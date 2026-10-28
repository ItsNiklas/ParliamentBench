import json
from random import shuffle
from typing import Optional, TypeVar, Type
import typing

from .game_primitives import (
    FascistPolicy,
    FascistRole,
    LeaderRole,
    LiberalPolicy,
    LiberalRole,
    Policy,
    Role,
    Vote,
    Ja,
    Nein,
    FASCIST_POLICIES,
    LIBERAL_POLICIES,
    PLAYERS,
)
from .game_logging import logger, format_state_for_display

if typing.TYPE_CHECKING:
    from .players import BasePlayer


# Type hint helper for class methods
T = TypeVar("T", bound="GameState")


class GameState:
    """Storage object for game state"""

    def __init__(self, playercount: int = 0) -> None:
        self.liberal_track = 0
        self.fascist_track = 0
        self.failed_votes = 0
        self.president: Optional["BasePlayer"] = None
        self.ex_president: Optional["BasePlayer"] = None
        self.chancellor: Optional["BasePlayer"] = None
        self.most_recent_policy = None
        self.last_votes: list[Vote] = []
        self.players: list["BasePlayer"] = []
        self.turn_count = 0  # Added turn_count property to GameState
        self.game_log: list[str] = []  # Added game_log from ParliamentGame
        self.chat_log: list[str] = []  # Added chat_log from ParliamentGame
        self.hitler: Optional["BasePlayer"] = None  # Added hitler from ParliamentGame
        self.game_data_logs: list[dict] = []  # Added structured logs for JSON summary
        self.investigated: list["BasePlayer"] = []  # OFFICIAL RULES (F13): nobody twice
        self.veto_proposed_by: Optional["BasePlayer"] = None  # OFFICIAL RULES (F10)

        # Former HitlerBoard attributes
        self.num_players = playercount
        if playercount > 0:
            self.num_liberals, self.num_fascists, self.fascist_track_actions = PLAYERS[
                playercount
            ]

            # Initialize policy deck
            self.policies: list[Policy] = [LiberalPolicy()] * LIBERAL_POLICIES + [
                FascistPolicy()
            ] * FASCIST_POLICIES
            shuffle(self.policies)
            self.discards: list[Policy] = []

    def to_json(self) -> str:
        """Convert the game state to a JSON string"""
        # This is an empty implementation for now
        # Will need to handle serializing objects like players, president, etc.
        state_dict = {
            "liberal_track": self.liberal_track,
            "fascist_track": self.fascist_track,
            "failed_votes": self.failed_votes,
            # Add other serializable fields as needed
        }
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls: Type[T], json_str: str, cutoff_rounds: int = 0) -> T:
        """Create a GameState instance from a JSON string
        
        Args:
            json_str: JSON string containing game data
            cutoff_rounds: Optional number of rounds to cut off from the end (default: 0)
        
        Returns:
            GameState instance with data from the JSON
        """
        data = json.loads(json_str)

        # Check if AvalonSH object is not null and exit if so
        if data.get("gameSetting", {}).get("avalonSH") is not None:
            print("AvalonSH object is not supported in this context")
            exit(1)
        
        # Create a new GameState instance
        state = cls(len(data["players"]))

        # Get the logs, cutting off the specified number of rounds if needed
        logs = data["logs"]
        if cutoff_rounds > 0:
            logs = logs[:-cutoff_rounds] if len(logs) > cutoff_rounds else []
        
        latest_log = logs[-1] if logs else None

        # Set track values from final state
        liberal_policies = sum(
            1 for log in logs if log.get("enactedPolicy") == "liberal"
        )
        fascist_policies = sum(
            1 for log in logs if log.get("enactedPolicy") == "fascist"
        )
        state.liberal_track = liberal_policies
        state.fascist_track = fascist_policies

        # Set turn count based on number of logs/turns
        state.turn_count = len(logs)

        # Set failed votes - count consecutive logs at the end with no enactedPolicy
        state.failed_votes = 0
        for log in reversed(logs):
            if "enactedPolicy" not in log:
                state.failed_votes += 1
            else:
                break

        # Veto power is automatically available when fascist_track >= 5
        # No need to store separate veto state

        # Set the most recent policy
        for log in reversed(logs):
            if log.get("enactedPolicy"):
                state.most_recent_policy = (
                    LiberalPolicy()
                    if log["enactedPolicy"] == "liberal"
                    else FascistPolicy()
                )
                break

        # Reconstruct policy deck from deckState in the latest log
        if latest_log and "deckState" in latest_log:
            state.policies = []

            for policy_type in latest_log["deckState"]:
                if policy_type == "liberal":
                    state.policies.append(LiberalPolicy())
                else:
                    state.policies.append(FascistPolicy())

            # Fill up the discards randomly to match the correct amounts
            state.discards = []
            missing_policies = FASCIST_POLICIES + LIBERAL_POLICIES - len(state.policies)
            if missing_policies > 0:
                state.discards.extend(
                    [LiberalPolicy()]
                    * (
                        LIBERAL_POLICIES
                        - sum(isinstance(p, LiberalPolicy) for p in state.policies)
                    )
                    + [FascistPolicy()]
                    * (
                        FASCIST_POLICIES
                        - sum(isinstance(p, FascistPolicy) for p in state.policies)
                    )
                )
                shuffle(state.discards)

        # Set the last votes if available
        if latest_log and "votes" in latest_log:
            state.last_votes = [Ja() if v else Nein() for v in latest_log["votes"]]

        # Create players from the players array
        from .players import RulePlayer

        player_data = data["players"]
        game_log = []  # Would normally extract from logs
        chat_log = []  # Would normally extract from logs

        # Create player objects
        for i, player_info in enumerate(player_data):
            # Convert role string to Role object
            from .game_primitives import LiberalRole, FascistRole, LeaderRole

            role_str = player_info["role"]
            if role_str == "liberal":
                role = LiberalRole()
            elif role_str == "fascist":
                role = FascistRole()
            elif role_str == "hitler":
                role = LeaderRole()
            else:
                raise ValueError(f"Unknown role: {role_str}")

            player = RulePlayer(
                id=i,
                name=player_info["username"],
                role=role,
                state=state,
                game_log=game_log,
                chat_log=chat_log,
            )
            state.players.append(player)
            
            # Set hitler player reference on the state
            if role_str == "hitler":
                state.hitler = player

        # Set knowledge for fascists and Hitler
        fascists = []
        hitler = None

        for player in state.players:
            if player.is_hitler:
                hitler = player
            elif player.is_fascist:
                fascists.append(player)

        # Make sure hitler is properly set in case it wasn't set earlier
        if hitler and state.hitler is None:
            state.hitler = hitler

        # Fascists know each other and Hitler
        for player in state.players:
            if player.is_fascist or (player.is_hitler and len(state.players) <= 6):
                player.fascists = fascists
                player.hitler = hitler

        # Set president/chancellor based on the latest log
        if latest_log:
            president_id = latest_log.get("presidentId")
            chancellor_id = latest_log.get("chancellorId")

            if president_id is not None and president_id < len(state.players):
                state.president = state.players[president_id]

            if chancellor_id is not None and chancellor_id < len(state.players):
                state.chancellor = state.players[chancellor_id]

        # Set ex-president from the second latest log if available
        if len(logs) > 1:
            second_latest_log = logs[-2]
            ex_president_id = second_latest_log.get("presidentId")

            if ex_president_id is not None and ex_president_id < len(state.players):
                state.ex_president = state.players[ex_president_id]

        logger.debug(f"Player List: {state.players}")
        logger.debug(f"Game State: {state}")
        logger.debug(f"Hitler player: {state.hitler.name if state.hitler else 'None'}")
        return state

    def shuffle_roles(self) -> list[Role]:
        """Shuffle and return player roles"""
        roles: list[Role] = (
            [LiberalRole()] * self.num_liberals
            + [FascistRole()] * self.num_fascists
            + [LeaderRole()]
        )
        shuffle(roles)
        return roles

    def draw_policy(self, num: int) -> list[Policy]:
        if len(self.policies) < num:
            # Shuffle the discards and add them to the remaining policies
            shuffle(self.discards)
            self.policies.extend(self.discards)
            self.discards = []

        if len(self.policies) < num:
            raise ValueError("Not enough policies in the deck?")

        drawn = self.policies[:num]
        self.policies = self.policies[num:]
        return drawn

    def discard(self, cards: list[Policy]) -> None:
        self.discards.extend(cards)

    def return_policy(self, policies: list[Policy]) -> None:
        self.policies.extend(policies)
        shuffle(self.policies)

    def eligible_chancellors(self, president: Optional["BasePlayer"]) -> list["BasePlayer"]:
        """OFFICIAL RULES (F11): the players `president` may nominate — everyone alive but the
        President, the last elected Chancellor, and the last elected President while more than five
        players are alive. At nomination time the engine keeps the last elected government in
        `chancellor` / `ex_president` (F5). Every player type and the engine use this one rule
        (there used to be four different versions, one counting the players at the start, not the living)."""
        alive = [p for p in self.players if not p.is_dead]
        return [p for p in alive
                if p is not president and p is not self.chancellor
                and not (len(alive) > 5 and p is self.ex_president)]

    def eligible_investigation_targets(self, president: Optional["BasePlayer"]) -> list["BasePlayer"]:
        """OFFICIAL RULES (F13): any other living player — but no player may be investigated twice."""
        return [p for p in self.players if not p.is_dead and p is not president and p not in self.investigated]

    def reshuffle_if_low(self) -> None:
        """OFFICIAL RULES (F7): at the end of a legislative session, fewer than three policies in
        the draw pile are shuffled together with the discard pile into a new draw pile. The deck used to be
        refilled only when a draw ran short, and kept the leftover cards on top, unshuffled."""
        if len(self.policies) < 3:
            self.policies = self.policies + self.discards
            self.discards = []
            shuffle(self.policies)

    def enact_policy(self, policy: Policy) -> bool:
        """
        :returns: Whether the policy enables a power
        """
        if policy.type == "liberal":
            self.liberal_track += 1
            self.most_recent_policy = policy

            return False
        else:
            self.fascist_track += 1
            self.most_recent_policy = policy

            # Veto power is automatically available when fascist_track >= 5
            # No need to set separate veto flag

            # Return whether an action should be taken
            if self.fascist_track <= len(self.fascist_track_actions):
                return self.fascist_track_actions[self.fascist_track - 1] is not None

            return False
        
    def calculate_gamestate_score(self, curr_log: dict, prev_log: dict) -> float:
        """Calculate a heuristic score for the current game state using stateeval.py
        
        Args:
            curr_log: Current turn log entry containing votes, policies, etc.
            prev_log: Previous turn log entry (unused for now)
            
        Returns:
            float: Score in [-1, 1] where positive favors Liberals, negative favors Fascists
        """
        # Import here to avoid circular imports
        from .metric.stateeval import evaluate_gamestate
        
        # Build gamestate dictionary for evaluation
        gamestate = {
            "liberal_policies": self.liberal_track,
            "fascist_policies": self.fascist_track,
            "deck": {
                "L": sum(1 for p in self.policies if p.type == "liberal"),
                "F": sum(1 for p in self.policies if p.type == "fascist")
            },
            "president": self.president.id if self.president else None,
            "round": self.turn_count,
            "unlocked_powers": self._get_unlocked_powers(),
            "president_role": self._get_president_role(),
            "num_players": self.num_players,
            "role_guesses_by_liberals": self._extract_role_guesses_from_log(prev_log if prev_log else dict())
        }
        
        # Create true_roles mapping for ground truth evaluation
        true_roles = {}
        for player in self.players:
            true_roles[player.id] = player.role.role
        
        return evaluate_gamestate(gamestate, true_roles)
    
    def _get_unlocked_powers(self) -> list[str]:
        """Get list of unlocked presidential powers based on fascist track"""
        if self.fascist_track <= 0 or self.fascist_track > len(self.fascist_track_actions):
            return []
        
        powers = []
        for i in range(self.fascist_track):
            if i < len(self.fascist_track_actions) and self.fascist_track_actions[i] is not None:
                action = self.fascist_track_actions[i]
                # Map game action names to stateeval power names
                if action == "kill":
                    powers.append("execution")
                elif action == "inspect":
                    powers.append("investigate")
                elif action == "policy":
                    powers.append("policy_peek")
                elif action == "choose":
                    # Note: "choose" power allows choosing the next president
                    # This doesn't directly map to stateeval power names, but we'll use a generic name
                    powers.append("choose_president")
        
        return powers
    
    def _get_president_role(self) -> str:
        """Get the true role of the current president"""
        if not self.president:
            return "liberal"  # Default assumption
        return self.president.role.role
    
    def _extract_role_guesses_from_log(self, curr_log: dict) -> dict:
        """Extract role guesses by liberal players from rapid assessments in the log
        
        Args:
            curr_log: Current turn log entry
            
        Returns:
            dict: {liberal_player_id: {target_player_id: "liberal"|"fascist"|"hitler"}}
        """
        role_guesses = {}
        
        # Get rapid assessments from current log
        rapid_assessments = curr_log.get("rapidAssessments", {})
        
        for player_id, assessment_text in rapid_assessments.items():
            # Only include guesses from liberal players
            player = next((p for p in self.players if p.id == player_id), None)
            if not player or player.role.role != "liberal":
                continue
            
            # Parse assessment text to extract role guesses
            player_guesses = {}
            if assessment_text:
                # Look for patterns like "PlayerName: Role" in the assessment
                import re
                lines = assessment_text.strip().split('\n')
                for line in lines:
                    # Match patterns like "Player0: Liberal", "Player1: Fascist", etc.
                    match = re.match(r'^([^:]+):\s*(Liberal|Fascist|Hitler|Unknown)', line.strip(), re.IGNORECASE)
                    if match:
                        target_name = match.group(1).strip()
                        role_guess = match.group(2).strip().lower()
                        
                        # Find the target player by name
                        target_player = next((p for p in self.players if p.name == target_name), None)
                        if target_player and role_guess in ["liberal", "fascist", "hitler"]:
                            player_guesses[target_player.id] = role_guess
            
            if player_guesses:
                role_guesses[player_id] = player_guesses
        
        return role_guesses

    @classmethod
    def parse_chat_json(cls, json_file_path: str, cutoff_rounds: int = 0) -> tuple[list[dict], list[str]]:
        """Parse chat messages from a JSON file and return structured chat and game logs
        
        Args:
            json_file_path: Path to the JSON file containing chat data
            cutoff_rounds: Optional number of rounds to cut off from the end (default: 0)
            
        Returns:
            tuple: (chat_log, game_log) lists containing formatted messages
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Check if the structure matches expected format
            if isinstance(data, list) and len(data) >= 2 and data[0] == "replayGameData" and isinstance(data[1], dict):
                chats = data[1].get("chats", [])
                
                # Create separate lists for chat and game logs
                chat_messages: list[dict] = []  # each: {userName, chat, state="unknown"}
                game_messages = []
                
                # Process all messages with timestamps
                all_messages = []
                
                for chat_entry in chats:
                    if not chat_entry:  # Skip empty entries
                        continue
                    
                    timestamp = chat_entry.get("timestamp", "")
                    if not timestamp:
                        continue
                        
                    username = chat_entry.get("userName", "System")
                    
                    # Game chat notification (goes to game_log)
                    if chat_entry.get("gameChat") is True and isinstance(chat_entry.get("chat"), list):
                        chat_texts = [item.get("text", "") for item in chat_entry["chat"] if item]
                        if chat_texts:
                            message = f"GAME: {''.join(chat_texts)}"
                            all_messages.append({
                                "timestamp": timestamp, 
                                "message": message, 
                                "type": "game",
                                # Flag to identify round starts (President nominates messages)
                                "is_round_start": ''.join(chat_texts).startswith("President") and "nominates" in ''.join(chat_texts)
                            })
                    
                    # Policy claim or other structured chat (goes to game_log)
                    elif chat_entry.get("isClaim") is True and isinstance(chat_entry.get("chat"), list):
                        chat_texts = []
                        for item in chat_entry["chat"]:
                            if item.get("text"):
                                # chat_texts.append(item["text"])
                                pass
                            elif item.get("claim"):
                                chat_texts.append(item["claim"])
                        
                        if chat_texts:
                            message = f"{username} claims: {''.join(chat_texts)}"
                            all_messages.append({
                                "timestamp": timestamp, 
                                "message": message, 
                                "type": "game",
                                "is_round_start": False
                            })
                    
                    # Regular chat message (goes to chat_log)
                    elif isinstance(chat_entry.get("chat"), str):
                        # Store user chat content separately; we'll format later
                        all_messages.append({
                            "timestamp": timestamp,
                            "userName": username,
                            "chat": chat_entry["chat"],
                            "type": "chat",
                            "is_round_start": False
                        })
                
                # Sort all messages by timestamp
                all_messages.sort(key=lambda x: x["timestamp"])
                
                # If cutoff_rounds > 0, we need to identify round starts and filter accordingly
                if cutoff_rounds > 0:
                    # Find indices of round starts
                    round_indices = [i for i, msg in enumerate(all_messages) if msg.get("is_round_start", False)]
                    
                    # If we have enough rounds to cut off
                    if len(round_indices) >= cutoff_rounds:
                        cutoff_index = round_indices[-(cutoff_rounds)]
                        all_messages = all_messages[:cutoff_index]
                
                # Separate into chat_log and game_log
                for msg in all_messages:
                    if msg["type"] == "chat":
                        chat_messages.append({
                            "userName": msg.get("userName", ""),
                            "chat": msg.get("chat", ""),
                            "state": "unknown"
                        })
                    else:
                        game_messages.append(msg["message"])
                
                return chat_messages, game_messages
            else:
                logger.error("Chat JSON has unexpected format")
                return [], []
        except Exception as e:
            logger.error(f"Error parsing chat JSON: {e}")
            return [], []

    def __str__(self) -> str:
        return format_state_for_display(self)
