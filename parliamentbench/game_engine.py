import os
import random
import sys
import argparse
import logging
import json
import datetime
import uuid
import concurrent.futures
from random import randint

from .config_loader import Config
from .game_state import GameState
from .players import BasePlayer, LLMPlayer, BasicLLMPlayer, RulePlayer, CPUPlayer, RandomPlayer
from .game_primitives import (
    FASCIST_POLICIES_TO_WIN,
    LIBERAL_POLICIES_TO_WIN,
    PLAYER_NAMES,
    Ja,
    Nein,
)
from .game_logging import *


class ParliamentGame:
    def __init__(self, args, config: Config = None) -> None:
        """Initialize the game

        Args:
            args: Parsed command line arguments
            config: Configuration object (optional, loaded from file or env)
        """
        # Store or create config
        self.config = config if config else Config()
        self.args = args
        
        # Store summary path from args or config
        self.summary_path = args.summary_path if hasattr(args, 'summary_path') else self.config.summary_path
        
        # Store forced player 1 role if provided
        self.role = args.role if hasattr(args, 'role') else None
        
        # Store player types configuration from args or config
        if hasattr(args, 'player_types') and args.player_types:
            self.player_types = args.player_types
        else:
            self.player_types = self.config.player_types if self.config.player_types else None
        
        # Store player configuration for summary
        self.player_config = []
        self.enable_pre_vote_discussion = self.config.enable_pre_vote_discussion
        self.enable_post_policy_discussion = self.config.enable_post_policy_discussion
        self.enable_reflection = self.config.enable_reflection
        self.enable_rapid_assessment = self.config.enable_rapid_assessment
        self.max_tokens = self.config.max_tokens
        self.request_timeout = self.config.request_timeout
        self.max_api_retries = self.config.max_api_retries
        
        # Store game end information for summary
        self.game_end_reason = None
        self.policy_counts_at_end = {"liberal": 0, "fascist": 0}
        self.summary_file_path = None
        
        # Handle JSON file if provided
        if hasattr(args, "gamestate_json") and args.gamestate_json:
            logger.info(f"Loading game state from JSON file: {args.gamestate_json}")
            try:
                with open(args.gamestate_json, "r") as f:
                    json_data = f.read()
                self.state = GameState.from_json(json_data, args.cutoff_rounds)
                self.playernum = len(self.state.players)
                logger.debug(
                    f"Game state loaded with {len(self.state.players)} players"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load game state from {args.gamestate_json}: {e}"
                )
                # Fall back to normal initialization
                logger.info(
                    f"Falling back to normal initialization with {args.players} players"
                )
                self.playernum = args.players
                self.state = GameState(args.players)
        else:
            logger.debug(f"Initializing new game with {args.players} players")
            self.playernum = args.players
            self.state = GameState(args.players)
        
        # Load chat data from JSON file if provided
        if hasattr(args, "chat_json") and args.chat_json:
            logger.info(f"Loading chat data from JSON file: {args.chat_json}")
            try:
                chat_messages, game_messages = GameState.parse_chat_json(args.chat_json, args.cutoff_rounds)
                
                if chat_messages or game_messages:
                    total_messages = len(chat_messages) + len(game_messages)
                    logger.info(f"Loaded {total_messages} messages ({len(chat_messages)} chat, {len(game_messages)} game)")
                    
                    # Add chat messages to chat_log
                    self.state.chat_log.extend(chat_messages)
                    # Normalize any legacy string entries (defensive, though new parser returns dicts)
                    normalized = []
                    for entry in self.state.chat_log:
                        if isinstance(entry, dict):
                            normalized.append(entry)
                        elif isinstance(entry, str):
                            if ':' in entry:
                                user, msg = entry.split(':', 1)
                                normalized.append({
                                    "userName": user.strip(),
                                    "chat": msg.strip(),
                                    "state": "unknown"
                                })
                            else:
                                normalized.append({
                                    "userName": "", "chat": entry, "state": "unknown"
                                })
                        else:
                            # Unexpected type; skip
                            continue
                    self.state.chat_log = normalized
                    
                    # Add game messages to game_log
                    if game_messages:
                        game_block = "Game Log:\n" + "\n".join(game_messages) + "\n" + "-" * 50 + "\n"
                        self.state.game_log.append(game_block)
                else:
                    logger.warning("No chat messages found in the provided JSON file")
            except Exception as e:
                logger.error(f"Failed to load chat data from {args.chat_json}: {e}")
                logger.info("Game will continue without chat data")

    def play(self) -> int:
        """Main game loop"""
        display_game_header()
        logger.info("Starting game")
        self.assign_players()
        self.inform_fascists()
        # self.choose_first_president()

        # Silently check win conditions
        if self.policy_win():
            logger.info("Policy win condition met")
            return 0

        if self.state.hitler and self.state.hitler.is_dead:
            logger.info("Hitler is dead - liberals win")
            return True

        while True:
            display_turn_header(self.state.turn_count + 1)
            self.state.turn_count += 1
            logger.info(f"Starting turn {self.state.turn_count}")
            self.log = "Turn " + str(self.state.turn_count) + "\n"
            self.vote_count = [0, 0]

            # Show current game state
            display_game_status(self.state)

            done = self.turn()

            self.log += "-----------------------------------\n"
            # Store but don't print the log and chat
            self.state.game_log.append(self.log)

            if done:
                logger.info("Game ending condition reached")
                break

            if self.state.turn_count >= 20:
                logger.warning("Ending game forcefully.")
                break

        return self.finish_game()

    def turn(self) -> bool:
        """
        Take a turn.
        """
        # First, pass on the presidency
        logger.debug("Updating presidency")
        self.state.ex_president = self.state.president
        self.set_next_president()

        # Initialize a log entry for this turn
        self.current_log_entry = {
            "_id": str(uuid.uuid4())[:24],
            "deckState": [p.type for p in self.state.policies],
            "votes": [],  # Will be filled after voting
            "presidentId": self.state.president.id if self.state.president else None,
        }

        # Calculate gamestate score for logging
        self.current_log_entry["gameStateScore"] = self.state.calculate_gamestate_score(self.current_log_entry, self.state.game_data_logs[-1] if self.state.game_data_logs else None)
        logger.debug(f"Calculated gamestate score: {self.current_log_entry['gameStateScore']}")
        
        # Ask the president to nominate a chancellor
        logger.debug(f"President {self.state.president} is nominating a chancellor")
        self.state.chancellor = self.nominate_chancellor()
        
        # Add chancellor to log entry
        self.current_log_entry["chancellorId"] = self.state.chancellor.id if self.state.chancellor else None

        # DISCUSSION PHASE (pre-vote) -------------------------------------------------
        if self.enable_pre_vote_discussion:
            logger.debug(
                f"Starting pre-vote discussion with {len(self.state.players)} players"
            )
            chat = "Discussion Before Voting:\n"
            # Only living players should participate in discussion
            living_discussants = [p for p in self.state.players if not p.is_dead]
            random.shuffle(living_discussants)
            logger.info("[bold green]Players discussing before vote...[/bold green]")
            for player in living_discussants:
                response = player.discuss(chat, "discussion_on_potential_government")
                self.state.chat_log.append({
                    "userName": player.name,
                    "chat": response,
                    "state": "discussion_on_potential_government",
                })
                chat += f"{str(player)}: {response}\n\n"
                logger.debug(f"Player {player.name} has provided their discussion input")

        # VOTING ---------------------------------------------------------------------
        voted = self.voting()

        living_players = [p for p in self.state.players if not p.is_dead]
        # 🆕 New reflection step after vote passes
        if self.enable_reflection and living_players:
            logger.info("Players privately reflecting")
            display_info_message("Players privately reflecting")
            # Parallel reflection for all living players
            reflections = self._parallel_reflect(living_players)
            
            # Store reflections in log entry
            reflections_data = {}
            for i, player in enumerate(living_players):
                monologue = reflections[i]
                reflections_data[player.id] = monologue
                display_player_inner_monologue(player, monologue, f"{player}'s inner monologue")
                logger.info(f"{player.name} reflection:\n{monologue}")
            
            self.current_log_entry["reflections"] = reflections_data
        
        # Rapid role assessment for each player
        if self.enable_rapid_assessment and living_players:
            logger.info("Players making rapid role assessments")
            display_info_message("Players making rapid role assessments")
            # Parallel rapid assessment for all living players
            assessments = self._parallel_rapid_assessment(living_players)
            
            # Store rapid assessments in log entry
            rapid_assessments_data = {}
            for i, player in enumerate(living_players):
                assessment = assessments[i]
                if assessment:  # Only log non-empty assessments
                    rapid_assessments_data[player.id] = assessment
                    display_player_inner_monologue(player, assessment, f"{player}'s rapid role assessment")
                    logger.info(f"{player.name} rapid assessment:\n{assessment}")
            
            self.current_log_entry["rapidAssessments"] = rapid_assessments_data

        if not voted:
            display_vote_results(self.vote_count, False)
            logger.info(
                f"Vote failed: {self.vote_count[0]} JA, {self.vote_count[1]} NEIN"
            )
            self.log += f"Vote failed: {self.vote_count[0]} voted JA to {self.vote_count[1]} voted NEIN\n"
            action_enacted = self.vote_failed()
        else:
            display_vote_results(self.vote_count, True)
            logger.info(
                f"Vote passed: {self.vote_count[0]} JA, {self.vote_count[1]} NEIN"
            )
            self.log += f"Vote passed: {self.vote_count[0]} voted JA to {self.vote_count[1]} voted NEIN\n"

            # Possibility to win if Hitler is chancellor and more than 2 fascist policies enacted.
            if self.hitler_chancellor_win():
                self.state.game_data_logs.append(self.current_log_entry)
                logger.info("Hitler chancellor win condition met.")
                return True

            action_enacted = self.vote_passed()

        # Post-policy discussion -----------------------------------------------------
        if action_enacted and self.enable_post_policy_discussion:
            # DISCUSS HERE
            logger.debug("Starting post-policy discussion")
            chat = "Discussion After Policy Enactment:\n"
            living_discussants = [p for p in self.state.players if not p.is_dead]
            random.shuffle(living_discussants)
            logger.info("[bold green]Players discussing after policy...[/bold green]")
            for player in living_discussants:
                response = player.discuss(chat, "after_policy")
                self.state.chat_log.append({
                    "userName": player.name,
                    "chat": response,
                    "state": "after_policy",
                })
                chat += f"{str(player)}: {response}\n\n"
                logger.debug(
                    f"Player {player.name} has provided their post-policy input"
                )

            self.perform_vote_action()

        if self.policy_win():
            logger.info("Policy win condition met")
            return True

        if self.state.hitler and self.state.hitler.is_dead:
            logger.info("Hitler is dead - liberals win")
            return True

        return False

    def assign_players(self) -> None:
        if self.state.players:
            logger.debug("Players already have roles assigned, skipping assignment")
            return
        logger.debug(f"Assigning roles to {self.playernum} players")
        roles = self.state.shuffle_roles()
        
        # If forcing player 1 role, swap the first role with the requested one
        if self.role:
            from .game_primitives import LiberalRole, FascistRole, LeaderRole
            forced_role_map = {
                'liberal': LiberalRole,
                'fascist': FascistRole,
                'hitler': LeaderRole
            }
            if self.role.lower() in forced_role_map:
                forced_role_class = forced_role_map[self.role.lower()]
                # Find the first instance of the forced role in the list
                for i, role in enumerate(roles):
                    if isinstance(role, forced_role_class):
                        # Swap with position 0
                        roles[0], roles[i] = roles[i], roles[0]
                        logger.info(f"Forced player 1 (Alice) to be {self.role}")
                        break

        display_info_message("[yellow]Assigning player roles...")

        for num in range(self.playernum):
            name = PLAYER_NAMES[num]
            # Determine player type from config or default (first player is LLM)
            if self.player_types and num < len(self.player_types):
                player_type = self.player_types[num].upper()
            else:
                player_type = "LLM" if num == 0 else "CPU"
            
            # Get LLM endpoint for this player type and index
            api_key, base_url, model = self.config.get_llm_endpoint(player_type, num)
            
            if player_type == "LLM":
                player = LLMPlayer(
                    num, name, roles.pop(0), self.state, self.state.game_log, self.state.chat_log, 
                    player_index=num, api_key=api_key, base_url=base_url, model=model, max_tokens=self.max_tokens, request_timeout=self.request_timeout, max_api_retries=self.max_api_retries
                )
            elif player_type == "BASICLLM":
                player = BasicLLMPlayer(
                    num, name, roles.pop(0), self.state, self.state.game_log, self.state.chat_log,
                    player_index=num, api_key=api_key, base_url=base_url, model=model, max_tokens=self.max_tokens, request_timeout=self.request_timeout, max_api_retries=self.max_api_retries
                )
            elif player_type == "CPU":
                player = CPUPlayer(
                    num, name, roles.pop(0), self.state, self.state.game_log, self.state.chat_log,
                    api_key=api_key, base_url=base_url, model=model, max_tokens=self.max_tokens, request_timeout=self.request_timeout, max_api_retries=self.max_api_retries
                )
            elif player_type == "RULE":
                player = RulePlayer(
                    num, name, roles.pop(0), self.state, self.state.game_log, self.state.chat_log,
                    api_key=api_key, base_url=base_url, model=model, max_tokens=self.max_tokens, request_timeout=self.request_timeout, max_api_retries=self.max_api_retries
                )
            elif player_type == "RANDOM":
                player = RandomPlayer(
                    num, name, roles.pop(0), self.state, self.state.game_log, self.state.chat_log,
                    api_key=api_key, base_url=base_url, model=model, max_tokens=self.max_tokens, request_timeout=self.request_timeout, max_api_retries=self.max_api_retries
                )
            else:
                raise ValueError(f"Unknown player type: {player_type}")

            # Store player configuration (will get model name later if needed)
            player_info = {
                "player_id": num,
                "name": name,
                "type": player_type,
                "base_url": base_url,
                "model": model,
            }
            self.player_config.append(player_info)

            if player.is_hitler:
                # Keep track of Hitler
                logger.debug(f"Player {name} is Hitler")
                self.state.hitler = player

            self.state.players.append(player)

    def inform_fascists(self) -> None:
        """
        Inform the fascists who the other fascists are.
        If there are 5-6 players, Hitler knows who the other fascists are.
        If there are 7+ players, Hitler does NOT know who the other fascists are.
        Regular fascists always know each other and Hitler in all game sizes.
        """
        logger.debug("Informing fascists of their team members")
        fascist_players = [player for player in self.state.players if player.is_fascist]
        non_hitler_fascists = [player for player in fascist_players if not player.is_hitler]

        for fascist in fascist_players:
            # Every fascist knows who Hitler is
            fascist.hitler = self.state.hitler
            
            if fascist.is_hitler:
                # Hitler only knows other fascists in 5-6 player games
                if self.playernum in [5, 6]:
                    logger.debug(f"Hitler {fascist.name} knows other fascists (5-6 players)")
                    fascist.fascists = non_hitler_fascists
                else:
                    logger.debug(f"Hitler {fascist.name} does NOT know other fascists (7+ players)")
                    fascist.fascists = []
            else:
                # Regular fascists always know each other and Hitler
                logger.debug(f"Fascist {fascist.name} knows all team members")
                fascist.fascists = fascist_players

    def choose_first_president(self) -> None:
        """
        Choose a random player to be the first president.
        """
        first_pres = self.state.players[randint(0, len(self.state.players) - 1)]
        self.state.president = first_pres
        logger.info(f"First president chosen: {first_pres}")

    def set_next_president(self) -> None:
        logger.debug("Setting next president")
        if self.state.president is None:
            self.choose_first_president()
            return

        self.state.president = self.state.players[
            (self.state.president.id + 1) % len(self.state.players)
        ]
        if self.state.president and self.state.president.is_dead:
            logger.debug(f"President {self.state.president} is dead, choosing next")
            self.set_next_president()
        else:
            logger.info(f"New president is {self.state.president}")

    def nominate_chancellor(self) -> BasePlayer:
        if self.state.president is None:
            logger.error("No president for chancellor nomination!")
            raise ValueError("No president!")

        # Get valid chancellor candidates
        valid_chancellors = [
            p for p in self.state.players
            if p != self.state.chancellor
            and p != self.state.president
            and not (self.playernum > 6 and p == self.state.ex_president)
            and not p.is_dead
        ]
        
        if not valid_chancellors:
            logger.error("No valid chancellors available!")
            raise ValueError("No valid chancellors!")

        chancellor = self.state.chancellor
        max_retries = 5
        retry_count = 0
        
        while (
            chancellor == self.state.chancellor
            or chancellor == self.state.president
            or (self.playernum > 6 and chancellor == self.state.ex_president)
            or (chancellor and chancellor.is_dead)
        ):
            if retry_count >= max_retries:
                logger.warning(f"President {self.state.president.name} failed to nominate valid chancellor after {max_retries} attempts. Selecting randomly from valid options: {[p.name for p in valid_chancellors]}")
                chancellor = random.choice(valid_chancellors)
                break
            
            logger.debug(f"Getting chancellor nomination from president (attempt {retry_count + 1}/{max_retries})...")
            chancellor = self.state.president.nominate_chancellor()
            retry_count += 1

        assert chancellor is not None
        display_chancellor_nomination(self.state.president, chancellor)
        return chancellor

    def voting(self) -> bool:
        """
        Get votes for the current pairing from all players.
        :returns: Whether the vote succeeded
        """
        logger.debug("Starting voting process")
        self.state.last_votes = []

        # Create a vote table for storing results
        vote_data = []
        vote_record = []  # For logging actual votes

        # Collect living players for parallel voting
        living_players = [player for player in self.state.players if not player.is_dead]
        
        if living_players:
            # Check if parallel processing is enabled
            from .players import BasePlayer
            if BasePlayer.enable_parallel_processing and len(living_players) > 1:
                # Parallel voting for all living players
                votes = self._parallel_vote_government(living_players)
            else:
                # Sequential voting (fallback)
                votes = []
                for player in living_players:
                    vote = player.vote_government()
                    votes.append(vote)
            
            # Process results in original player order
            vote_index = 0
            for player in self.state.players:
                if not player.is_dead:
                    vote = votes[vote_index]
                    vote_index += 1
                    self.state.last_votes.append(vote)
                    vote_data.append((player, vote))
                    vote_record.append(bool(vote))  # Convert Vote object to bool for JSON
                    # Display individual votes with proper formatting
                    display_player_vote(player, vote)
                else:
                    # Dead players can't vote
                    vote_record.append(None)

        # Add votes to the current log entry
        self.current_log_entry["votes"] = vote_record

        # Display a table of all votes
        display_vote_table(vote_data)

        # Log individual votes to self.log
        for player, vote in vote_data:
            self.log += f"{player.name} voted {"JA" if vote else "NEIN"}\n"

        positivity = 0

        for vote in self.state.last_votes:
            if vote:
                positivity += 1
                self.vote_count[0] += 1
            else:
                positivity -= 1
                self.vote_count[1] += 1

        return positivity > 0
    
    def _parallel_collect(self, players, method):
        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(players)) as executor:
            # Submit all tasks
            future_to_player = {executor.submit(method, player): player for player in players}
            
            # Collect results in order
            res = []
            for player in players:
                # Find the future for this player
                future = next(future for future, p in future_to_player.items() if p == player)
                vote = future.result()  # This will block until the result is ready
                res.append(vote)
        
        return res

    def _parallel_vote_government(self, players):
        logger.debug(f"Starting parallel voting for {len(players)} players")
        
        def vote_wrapper(player):
            try:
                return player.vote_government()
            except Exception as e:
                logger.error(f"Error in voting for player {player.name}: {e}")
                # Return random vote as fallback
                return random.choice([Ja(), Nein()])
        
        return self._parallel_collect(players, vote_wrapper)
    
    def _parallel_discuss(self, players, chat, stage):
        logger.debug(f"Starting parallel discussion for {len(players)} players")
        
        def discuss_wrapper(player_data):
            player, current_chat = player_data
            try:
                return player.discuss(current_chat, stage)
            except Exception as e:
                logger.error(f"Error in discussion for player {player.name}: {e}")
                return "I have nothing to say right now."
        
        # All players get the same initial chat context
        player_data_list = [(player, chat) for player in players]
        return self._parallel_collect(player_data_list, discuss_wrapper)

    def _parallel_reflect(self, players):
        logger.debug(f"Starting parallel reflection for {len(players)} players")
        
        def reflect_wrapper(player):
            try:
                return player.reflect_on_roles()
            except Exception as e:
                logger.error(f"Error in reflection for player {player.name}: {e}")
                return "I'm thinking about the current situation..."
        
        return self._parallel_collect(players, reflect_wrapper)

    def _parallel_rapid_assessment(self, players):
        logger.debug(f"Starting parallel rapid assessment for {len(players)} players")
        
        def assess_wrapper(player):
            try:
                if hasattr(player, 'rapid_role_assessment'):
                    return player.rapid_role_assessment()
                return ""
            except Exception as e:
                logger.error(f"Error in rapid assessment for player {player.name}: {e}")
                return ""
        
        return self._parallel_collect(players, assess_wrapper)

    def _handle_three_failed_votes(self) -> bool:
        """
        Handle the case when 3 failed votes have accumulated.
        Resets failed votes counter and automatically enacts the top policy.
        Returns the result of enacting the policy.
        """
        self.state.failed_votes = 0
        logger.warning("3 failed votes - enacting top policy automatically")
        auto_policy = self.state.draw_policy(1)[0]
        display_election_tracker_full()
        logger.info(f"Enacting policy automatically: {auto_policy}")
        display_policy_enacted(auto_policy)
        
        # Update log entry with the automatically enacted policy
        self.current_log_entry["enactedPolicy"] = auto_policy.type
        self.state.game_data_logs.append(self.current_log_entry)
        return self.state.enact_policy(auto_policy)

    def vote_failed(self) -> bool:
        logger.info("Vote failed")
        self.state.failed_votes += 1
        
        # Save the current log entry to the state's data logs
        # even though the vote failed
        self.current_log_entry["presidentHand"] = []
        self.current_log_entry["chancellorHand"] = []
        self.current_log_entry["presidentClaim"] = []
        self.current_log_entry["chancellorClaim"] = []
        self.current_log_entry["policyPeek"] = []
        self.current_log_entry["policyPeekClaim"] = []
        # Initialize empty reflection and assessment data for failed votes
        if "reflections" not in self.current_log_entry:
            self.current_log_entry["reflections"] = {}
        if "rapidAssessments" not in self.current_log_entry:
            self.current_log_entry["rapidAssessments"] = {}

        if self.state.failed_votes == 3:
            return self._handle_three_failed_votes()
        else:
            display_failed_votes(self.state.failed_votes)
            self.state.game_data_logs.append(self.current_log_entry)
            return False

    def vote_passed(self) -> bool:
        """
        The vote has passed! Get the president and chancellor to do their thing.
        """
        logger.info("Vote passed, proceeding with policy selection")
        if self.state.president is None or self.state.chancellor is None:
            logger.error("Missing president/chancellor!")
            raise ValueError()

        drawn_policies = self.state.draw_policy(3)
        logger.debug(f"Drew policies: {drawn_policies}")

        # Record the drawn policies in the log entry
        self.current_log_entry["presidentHand"] = [p.type for p in drawn_policies]

        display_info_message(f"[bold green]President selecting policies...[/bold green] ({[p.type[0] for p in drawn_policies]})")
        (take, discard) = self.state.president.filter_policies(drawn_policies)

        logger.debug(f"President kept {take}, discarded {discard}")
        self.state.discard([discard])

        # Record the chancellor's hand in the log entry
        self.current_log_entry["chancellorHand"] = [p.type for p in take]

        display_info_message(f"[bold green]Chancellor selecting policy...[/bold green] ({[p.type[0] for p in take]})")
        (enact, discard) = self.state.chancellor.enact_policy(take)

        logger.debug(f"Chancellor selected {enact}, would discard {discard}")

        # Check for veto power (available after 5 fascist policies)
        if self.state.fascist_track >= 5:
            logger.info("Veto power is available - checking for veto")
            display_info_message("[bold yellow]Veto power available - President and Chancellor deciding...[/bold yellow]")
            
            # Both president and chancellor must agree to veto
            logger.debug(f"President {self.state.president} considering veto...")
            president_veto = self.state.president.veto([enact, discard])
            logger.debug(f"President veto decision: {president_veto}")
            
            logger.debug(f"Chancellor {self.state.chancellor} considering veto...")
            chancellor_veto = self.state.chancellor.veto([enact, discard])
            logger.debug(f"Chancellor veto decision: {chancellor_veto}")
            
            if president_veto and chancellor_veto:
                logger.info("Veto enacted by both President and Chancellor")
                display_veto()
                self.log += f"Veto enacted by President {self.state.president} and Chancellor {self.state.chancellor}\n"
                
                # Discard both remaining policies
                self.state.discard([enact, discard])
                
                # Record veto in log entry
                self.current_log_entry["enactedPolicy"] = "veto"
                self.current_log_entry["vetoUsed"] = True
                
                # Advance the election tracker (increment failed votes)
                self.state.failed_votes += 1
                logger.info(f"Election tracker advanced to {self.state.failed_votes}")
                
                # Check if 3 failed votes triggers automatic policy enactment
                if self.state.failed_votes == 3:
                    return self._handle_three_failed_votes()
                else:
                    display_failed_votes(self.state.failed_votes)
                    self.state.game_data_logs.append(self.current_log_entry)
                    return False
            else:
                logger.info(f"Veto rejected - President: {president_veto}, Chancellor: {chancellor_veto}")
                display_info_message("[bold green]Veto rejected, proceeding with policy enactment...[/bold green]")

        # No veto, proceed with policy enactment
        self.state.discard([discard])

        display_policy_enacted(enact)

        # Record the enacted policy in the log entry
        self.current_log_entry["enactedPolicy"] = enact.type
        # Add the log entry to the state's data logs
        self.state.game_data_logs.append(self.current_log_entry)

        self.log += f"{str(enact)} policy enacted by President {self.state.president} and Chancellor {self.state.chancellor}\n"
        
        # Update CPU players' reputation after legislation
        for player in self.state.players:
            if isinstance(player, CPUPlayer) and not player.is_dead:
                player.update_after_legislation()
        
        return self.state.enact_policy(enact)

    def hitler_chancellor_win(self) -> bool:
        result = self.state.fascist_track >= 3 and self.state.chancellor == self.state.hitler
        if result:
            logger.info(
                "Hitler elected as chancellor with 3+ fascist policies - fascists win!"
            )
        return result

    def policy_win(self) -> bool:
        liberal_win = self.state.liberal_track == LIBERAL_POLICIES_TO_WIN
        fascist_win = self.state.fascist_track == FASCIST_POLICIES_TO_WIN
        if liberal_win:
            logger.info("Liberals win via policy track!")
        elif fascist_win:
            logger.info("Fascists win via policy track!")
        return liberal_win or fascist_win

    def perform_vote_action(self) -> None:
        # Check if we have a valid fascist track position
        if self.state.fascist_track <= 0 or self.state.fascist_track > len(self.state.fascist_track_actions):
            logger.debug(f"No action available for fascist track position {self.state.fascist_track}")
            return
            
        action = self.state.fascist_track_actions[self.state.fascist_track - 1]
        if action is None:
            logger.debug("No action for current fascist track position")
            return

        if not self.state.president:
            logger.warning("No president for vote action")
            return

        logger.info(f"Performing vote action: {action}")
        display_special_action(action)

        if action == "policy":
            top_three = self.state.draw_policy(3)
            logger.debug(f"President viewing policies: {top_three}")
            display_info_message(
                "[bold blue]President examining top policies...[/bold blue]"
            )
            self.state.president.view_policies(top_three)
            display_policy_view(self.state.president)
            self.state.return_policy(top_three)

        elif action == "kill":
            display_info_message(
                "[bold red]President choosing player to execute...[/bold red]"
            )
            # Get valid execution targets
            valid_targets = [
                p for p in self.state.players
                if not p.is_dead and p != self.state.president
            ]
            
            if not valid_targets:
                logger.warning("No valid execution targets available!")
                return
            
            killed_player = self.state.president.kill()
            logger.debug(f"Initial kill target: {killed_player}")
            
            max_retries = 5
            retry_count = 0
            while killed_player.is_dead or killed_player == self.state.president:
                if retry_count >= max_retries:
                    logger.warning(f"President {self.state.president.name} failed to select valid execution target after {max_retries} attempts. Selecting randomly from: {[p.name for p in valid_targets]}")
                    killed_player = random.choice(valid_targets)
                    break
                killed_player = self.state.president.kill()
                retry_count += 1
                
            killed_player.is_dead = True
            self.log += f"Player {killed_player} has been killed by President {self.state.president}\n"
            logger.info(f"Player {killed_player} has been killed")
            display_player_executed(killed_player, self.state.president)

        elif action == "inspect":
            display_info_message(
                "[bold yellow]President investigating a player...[/bold yellow]"
            )
            # Get valid inspection targets
            valid_targets = [
                p for p in self.state.players
                if not p.is_dead and p != self.state.president
            ]
            
            if not valid_targets:
                logger.warning("No valid inspection targets available!")
                return
            
            inspect = self.state.president.inspect_player()
            logger.debug(f"Initial inspect target: {inspect}")
            
            max_retries = 5
            retry_count = 0
            while inspect.is_dead or inspect == self.state.president:
                if retry_count >= max_retries:
                    logger.warning(f"President {self.state.president.name} failed to select valid inspection target after {max_retries} attempts. Selecting randomly from: {[p.name for p in valid_targets]}")
                    inspect = random.choice(valid_targets)
                    break
                inspect = self.state.president.inspect_player()
                retry_count += 1
                
            self.state.president.inspected_players = (
                f"{inspect.name} is a {inspect.role.party_membership}"
            )
            logger.info(f"President inspected {inspect}")
            display_player_investigated(self.state.president, inspect)
            
            # Update CPU players' knowledge about the investigated player
            for player in self.state.players:
                if isinstance(player, CPUPlayer) and not player.is_dead:
                    # Only the investigating president learns the truth
                    if player == self.state.president:
                        player.update_known_role_from_investigation(inspect.name, inspect.role.party_membership)

        elif action == "choose":
            display_info_message(
                "[bold cyan]President choosing next president...[/bold cyan]"
            )
            # Get valid next president candidates
            valid_candidates = [
                p for p in self.state.players
                if not p.is_dead and p != self.state.president
            ]
            
            if not valid_candidates:
                logger.warning("No valid presidential candidates available!")
                return
            
            current_president = self.state.president
            chosen = self.state.president
            logger.debug("President choosing next president")
            
            max_retries = 5
            retry_count = 0
            while chosen == self.state.president or chosen.is_dead:
                if retry_count >= max_retries:
                    logger.warning(f"President {self.state.president.name} failed to select valid next president after {max_retries} attempts. Selecting randomly from: {[p.name for p in valid_candidates]}")
                    chosen = random.choice(valid_candidates)
                    break
                chosen = self.state.president.choose_next()
                retry_count += 1

            self.state.president = chosen
            logger.info(f"President selected {chosen} as next president")
            self.log += f"President {current_president} has chosen {chosen} as the next president\n"
            display_next_president_chosen(current_president, chosen)

        else:
            logger.error(f"Unknown action: {action}")
            assert False, "Unrecognised action!"

    def finish_game(self) -> int:
        logger.info("Game finished, determining winner")

        # Store policy counts at game end
        self.policy_counts_at_end = {
            "liberal": self.state.liberal_track,
            "fascist": self.state.fascist_track
        }

        if self.hitler_chancellor_win():
            display_game_over("hitler_chancellor")
            logger.info("Fascists win by electing Hitler!")
            self.game_end_reason = "hitler_chancellor"
            result = -2
        elif self.policy_win():
            if self.state.liberal_track == LIBERAL_POLICIES_TO_WIN:
                display_game_over("liberal_policy")
                logger.info("Liberals win by policy!")
                self.game_end_reason = "liberal_policy"
                result = 1
            else:
                display_game_over("fascist_policy")
                logger.info("Fascists win by policy!")
                self.game_end_reason = "fascist_policy"
                result = -1
        elif self.state.hitler and self.state.hitler.is_dead:
            display_game_over("hitler_killed")
            logger.info("Liberals win by shooting Hitler!")
            self.game_end_reason = "hitler_killed"
            result = 2
        else:
            logger.warning("Game ended with no clear winner condition")
            self.game_end_reason = "no_clear_winner"
            result = 0
        
        # Print token usage summary
        if os.getenv("PARLIAMENTBENCH_DISABLE_TOKEN_SUMMARY") != "1":
            try:
                from .metric.token_tracker import get_tracker
                tracker = get_tracker()
                tracker.print_summary()
                tracker.save_summary()
            except Exception as e:
                logger.warning(f"Could not generate token usage summary: {e}")
        
        # Generate and write summary JSON file
        self.write_summary_json()
        
        return result
        
    def write_summary_json(self) -> None:
        """
        Generates and writes a summary JSON file after the game finishes.
        The file contains game settings, players, roles, and detailed logs.
        """
        logger.info("Generating game summary JSON")
        
        # Generate a unique game ID. The timestamp only has second resolution,
        # so a short random suffix is appended: benchmark runs write every game
        # into the same directory, and games that finish within the same second
        # would otherwise overwrite each other and be dropped from the report.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        if hasattr(self.args, "gamestate_json") and self.args.gamestate_json:
            stem = os.path.splitext(os.path.basename(self.args.gamestate_json.replace('_summary', '')))[0]
            game_id = f"{stem}_{timestamp}_{suffix}"
        else:
            game_id = f"Game_{timestamp}_{suffix}"
        
        # Collect model names from players without triggering network calls.
        # For non-LLM players we report N/A, and for LLM players we use
        # configured or already-cached model names.
        player_models = []
        for index, player in enumerate(self.state.players):
            player_type = None
            configured_model = None
            if index < len(self.player_config):
                player_type = str(self.player_config[index].get("type", "")).upper()
                configured_model = self.player_config[index].get("model")

            if player_type not in {"LLM", "BASICLLM"}:
                model_name = "N/A"
            elif configured_model:
                model_name = configured_model
            else:
                # Use the cached model name if available, but never probe API here.
                model_name = getattr(player, "_model_name", None) or "unknown"
            player_models.append(model_name)
        
        # Create summary structure
        summary = {
            "_id": game_id,
            "gameSetting": {
                "avalonSH": None,
                "rebalance6p": False,
                "rebalance7p": False,
                "rebalance9p": False,
                "casualGame": False,
                "practiceGame": False,
                "unlistedGame": False,
                "noTopdecking": 0
            },
            "date": datetime.datetime.now().isoformat() + "Z",
            "playerConfiguration": {
                "player_types": self.player_types,
                "player_config": self.player_config,
                "model_names": player_models
            },
            "gameEndReason": self.game_end_reason,
            "policyCountsAtEnd": self.policy_counts_at_end,
            "players": [],
            "libElo": {
                "overall": 0,
                "season": 0
            },
            "fasElo": {
                "overall": 0,
                "season": 0
            },
            "logs": [],
            "chats": [],
            "__v": 0
        }
        
        # Add player information
        for player in self.state.players:
            player_id = str(uuid.uuid4())[:24]  # Generate a random ID
            
            player_info = {
                "_id": player_id,
                "username": player.name,
                "role": "hitler" if player.is_hitler else "fascist" if player.is_fascist else "liberal",
                "icon": 0
            }
            summary["players"].append(player_info)
        
        # Use the structured game log data we've been collecting
        if self.state.game_data_logs:
            summary["logs"] = self.state.game_data_logs
        else:
            # Fallback to generate some placeholder data if no logs were collected
            logger.warning("No game logs were collected")
        
        # Add chat history from game
        if self.state.chat_log:
            for chat_entry in self.state.chat_log:
                if isinstance(chat_entry, dict):
                    normalized_chat = {
                        "chat": chat_entry.get("chat", ""),
                        "userName": chat_entry.get("userName", ""),
                        "state": chat_entry.get("state", "unknown")
                    }
                    summary["chats"].append(normalized_chat)
                elif isinstance(chat_entry, str):
                    normalized_chat = {
                        "chat": chat_entry,
                        "userName": "",
                        "state": "unknown",
                    }
                    summary["chats"].append(normalized_chat)
    
        # Write to a JSON file in the summaries directory
        summary_filename = f"{game_id}_summary.json"
        summary_path = self.summary_path
        
        # Create summaries directory if it doesn't exist
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
            
        with open(f"{summary_path}/{summary_filename}", "w") as f:
            json.dump(summary, f, indent=4)
        self.summary_file_path = f"{summary_path}/{summary_filename}"
        logger.info(f"Game summary written to {self.summary_file_path}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Secret Hitler Game Simulator")
    parser.add_argument(
        "--config",
        "-f",
        type=str,
        help="Path to YAML configuration file (overrides other arguments and env vars)"
    )
    parser.add_argument(
        "--players", "-p", type=int, default=5, help="Number of players (default: 5)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.txt",
        help="Output file path (default: output.txt)",
    )
    parser.add_argument(
        "--gamestate-json",
        "-g",
        type=str,
        help="Path to load game state from JSON file",
    )
    parser.add_argument(
        "--chat-json",
        "-c",
        type=str,
        help="Path to load chat messages from JSON file",
    )
    parser.add_argument(
        "--cutoff-rounds",
        "-n",
        type=int,
        default=0,
        help="Number of rounds to cut off from the end (default: 0)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        help="Use simple game mode (unused for now)",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="./runs",
        help="Path to write game summary files (default: ./runs)",
    )
    parser.add_argument(
        "--player-types",
        type=lambda s: s.split(','),
        default=None,
        help="Comma-separated list of player types (LLM or CPU) for each player (e.g., 'LLM,CPU,CPU,CPU,CPU'). Default: first player is LLM, rest are CPU",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["liberal", "fascist", "hitler"],
        default=None,
        help="Force player 1 (Alice) to have a specific role: liberal, fascist, or hitler",
    )
    args = parser.parse_args()

    # Load configuration - config file is now required
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info("No config file specified, using default configuration")
        config = Config()
    
    # Override config with command line arguments if provided
    if args.players != 5:  # 5 is default
        config.players = args.players
    if args.summary_path != "./runs":
        config.summary_path = args.summary_path
    if args.log_level != "INFO":
        config.log_level = args.log_level
    if args.player_types:
        config.player_types = args.player_types
    
    # Update args from config
    args.players = config.players
    args.summary_path = config.summary_path
    args.log_level = config.log_level
    if not args.player_types:
        args.player_types = config.player_types
    
    # Apply processing, theme and generation config
    from .runtime import apply_runtime_settings
    settings = apply_runtime_settings(config)

    # Set the log level based on command line argument or config
    log_level = getattr(logging, args.log_level)
    set_log_level(log_level)

    if settings["theme"]:
        logger.info(
            f"[theme] active theme = {settings['theme']} (strict={settings['theme_strict']})"
        )

    # Initialize the file logger with the output path from args
    init_file_logger(args.output)

    # Display start message and initialize game with all args
    display_game_start(args.players if not args.gamestate_json else 0)
    logger.info("Starting Hitler Game...")

    # Initialize game with args object and config
    game = ParliamentGame(args, config)
    game.play()

    sys.stdout = sys.__stdout__
