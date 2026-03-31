from random import choice, getrandbits
import random

from .base_player import BasePlayer
from ..game_primitives import Ja, Nein, Policy, Vote
from ..game_logging import logger, display_policy_table, display_player_discussion
from ..metric.token_tracker import track_response
from ..theme import apply_to_messages


class BasicLLMPlayer(BasePlayer):
    """
    Basic LLM Player - makes decisions using LLM but without advanced features:
    - No chain-of-thought reasoning prompts
    - No memory/inspection history
    - Simpler prompts focused on direct decisions
    """
    
    def __init__(self, id, name: str, role, state, game_log, chat_log, player_index: int = 0, api_key: str = None, base_url: str = None, model: str = None, max_tokens: int = 1024, request_timeout: int = 60, max_api_retries: int = 3) -> None:
        super(BasicLLMPlayer, self).__init__(
            id,
            name,
            role,
            state,
            game_log,
            chat_log,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_api_retries=max_api_retries,
        )
        # Override to disable memory tracking
        self.use_memory = False

    def get_basic_completion(self, prompt: str, _stage: str) -> str:
        """Get completion without memory/CoT features"""
        openai_model = self.get_model_name()

        # Prepare recent chat context
        recent_chat_entries = []
        for entry in self.state.chat_log[-10:]:  # Shorter context window
            if not isinstance(entry, dict):
                continue
            user = entry.get("userName", "")
            msg = entry.get("chat", "")
            msg_stripped = msg.strip()
            if msg_stripped.startswith('"') and msg_stripped.endswith('"'):
                msg_stripped = msg_stripped[1:-1]
            recent_chat_entries.append(f'{user}: "{msg_stripped}"')
        formatted_recent_chat = "\n".join(recent_chat_entries)

        # Simple system prompt without complex strategy instructions
        system_content = f"""You are playing Secret Hitler. 5 players: three Liberals, one Fascist, one Hitler.

YOUR NAME: {self.name}
YOUR ROLE: {self.role} {"(Fascist)" if self.role.role == "hitler" else ""}

{self.get_known_state()}"""
        
        # Simple prompt without memory
        full_prompt = f"""Recent game log:
{"\n".join(self.state.game_log[-10:])}

Recent discussions:
{formatted_recent_chat}

{prompt}"""

        base_msg = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": full_prompt},
        ]

        # Apply the loaded-terms theme (no-op when no theme is active).
        base_msg = apply_to_messages(base_msg)

        content = None
        for attempt in range(self.max_retries):
            # On retry, nudge the model to be terser and give it more room — the
            # usual cause of content=None is reasoning tokens exhausting
            # max_tokens before any visible answer is emitted. Append to the
            # (already-themed) user message so the theme survives the retry.
            msg = list(base_msg)
            if attempt > 0:
                msg[-1] = {
                    "role": "user",
                    "content": msg[-1]["content"]
                    + "\n\nBe as brief as possible. Answer in one short line.",
                }
            mt = self.basic_max_tokens * (attempt + 1)  # e.g. 512, 1024, 1536
            try:
                response = self.openai_client.chat.completions.create(
                    model=openai_model,
                    messages=msg,
                    max_tokens=mt,  # Shorter responses
                    temperature=0.7,
                    timeout=self.request_timeout,
                )
            except Exception as e:
                logger.warning(
                    f"API error for {self.name} at stage {_stage} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                import time
                time.sleep(2 ** attempt)
                continue

            track_response(response, stage=_stage, player_name=self.name)

            content = response.choices[0].message.content
            if content is not None:
                break
            logger.warning(
                f"LLM response is None for {self.name} at stage {_stage} "
                f"(attempt {attempt + 1}/{self.max_retries})"
            )

        if content is None:
            logger.error(
                f"LLM response is None after {self.max_retries} attempts for "
                f"{self.name} at stage {_stage}, returning empty string."
            )
            content = ""

        return content

    def vote_government(self) -> Vote:
        """Vote without CoT reasoning"""
        prompt = f"""Vote on the nominated government (President: {self.state.president}, Chancellor: {self.state.chancellor}).

Respond with ONLY "JA" (yes) or "NEIN" (no)."""

        response = self.get_basic_completion(prompt, "Vote")

        if "JA" in response.upper() and "NEIN" not in response.upper():
            return Ja()
        elif "NEIN" in response.upper():
            return Nein()

        logger.debug(f"{self.name}: No clear vote, returning random.")
        return random.choice([Ja(), Nein()])

    def nominate_chancellor(self) -> "BasePlayer":
        """Nominate chancellor without CoT"""
        # Get eligible players (not self, not current chancellor, not dead)
        eligible = [
            p for p in self.state.players 
            if p != self 
            and p != self.state.chancellor
            and not p.is_dead
        ]
        # Also exclude ex-president in larger games
        if len(self.state.players) > 6 and self.state.ex_president:
            eligible = [p for p in eligible if p != self.state.ex_president]
        
        if not eligible:
            logger.error("No eligible chancellors!")
            return choice(self.state.players)
        
        prompt = f"""Nominate a chancellor.
        
VALID OPTIONS (choose one of these names EXACTLY):
{', '.join([p.name for p in eligible])}

You cannot pick yourself, the current chancellor, dead players, or the previous president (in 7+ player games).

Respond with ONLY the player name from the valid options."""

        response = self.get_basic_completion(prompt, "Nominate Chancellor")

        # Try exact name match first
        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear nomination, returning random from eligible: {[p.name for p in eligible]}")
        return choice(eligible)

    def view_policies(self, policies: list[Policy]) -> None:
        """View policies without analysis"""
        display_policy_table(policies, f"Policies viewed by {self.name}", True)
        # No LLM reasoning needed for viewing

    def kill(self) -> "BasePlayer":
        """Execute a player without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]
        
        if not eligible:
            logger.error("No eligible players to execute!")
            return choice(self.state.players)

        prompt = f"""Execute a player.
        
VALID OPTIONS (choose one of these names EXACTLY):
{', '.join([p.name for p in eligible])}

You cannot execute yourself or dead players.

Respond with ONLY the player name from the valid options."""

        response = self.get_basic_completion(prompt, "Kill")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear execution choice, returning random from eligible: {[p.name for p in eligible]}")
        return choice(eligible)

    def inspect_player(self) -> "BasePlayer":
        """Inspect a player without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]
        
        if not eligible:
            logger.error("No eligible players to inspect!")
            return choice(self.state.players)

        prompt = f"""Inspect a player's party membership.
        
VALID OPTIONS (choose one of these names EXACTLY):
{', '.join([p.name for p in eligible])}

You cannot inspect yourself or dead players.

Respond with ONLY the player name from the valid options."""

        response = self.get_basic_completion(prompt, "Inspect")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear inspection choice, returning random from eligible: {[p.name for p in eligible]}")
        return choice(eligible)

    def choose_next(self) -> "BasePlayer":
        """Choose next president without CoT"""
        eligible = [p for p in self.state.players if p != self and not p.is_dead]
        
        if not eligible:
            logger.error("No eligible players to choose as next president!")
            return choice(self.state.players)

        prompt = f"""Choose the next president.
        
VALID OPTIONS (choose one of these names EXACTLY):
{', '.join([p.name for p in eligible])}

You cannot choose yourself or dead players.

Respond with ONLY the player name from the valid options."""

        response = self.get_basic_completion(prompt, "Choose President")

        for player in eligible:
            if player.name.upper() in response.upper():
                return player

        logger.debug(f"{self.name}: No clear president choice, returning random from eligible: {[p.name for p in eligible]}")
        return choice(eligible)

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """Chancellor policy choice without CoT"""
        display_policy_table(policies, f"Chancellor {self.name} selects policy", True)

        prompt = f"""You are chancellor. Choose which card to DISCARD:
Card 1: {str(policies[0])}
Card 2: {str(policies[1])}

Respond with "DISCARD: Card 1" or "DISCARD: Card 2"."""

        response = self.get_basic_completion(prompt, "Enact Policy")

        if "DISCARD: CARD 1" in response.upper():
            return (policies[1], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            return (policies[0], policies[1])

        logger.debug(f"{self.name}: No clear policy choice, returning random.")
        return (policies[0], policies[1]) if random.random() > 0.5 else (policies[1], policies[0])

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        """President policy filtering without CoT"""
        display_policy_table(policies, f"President {self.name} draws policies", True)

        prompt = f"""You are president. Choose which card to DISCARD:
Card 1: {str(policies[0])}
Card 2: {str(policies[1])}
Card 3: {str(policies[2])}

Respond with "DISCARD: Card 1", "DISCARD: Card 2", or "DISCARD: Card 3"."""

        response = self.get_basic_completion(prompt, "Filter Policies")

        if "DISCARD: CARD 1" in response.upper():
            return ([policies[1], policies[2]], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            return ([policies[0], policies[2]], policies[1])
        elif "DISCARD: CARD 3" in response.upper():
            return ([policies[0], policies[1]], policies[2])

        logger.debug(f"{self.name}: No clear filter choice, returning random.")
        return ([policies[0], policies[1]], policies[2])

    def veto(self, policies: list[Policy]) -> bool:
        """Veto decision without CoT"""
        prompt = f"""Consider vetoing these policies: {[p.type for p in policies]}
Current state: {self.state.liberal_track}L / {self.state.fascist_track}F enacted

Respond with ONLY "VETO" or "NO VETO"."""

        response = self.get_basic_completion(prompt, "Veto")

        if "NO VETO" in response.upper():
            return False
        elif "VETO" in response.upper():
            return True

        return bool(getrandbits(1))

    def discuss(self, chat: str, stage: str) -> str:
        """Discussion without complex strategy"""
        if stage == "discussion_on_potential_government":
            prompt = """It's discussion time about the proposed government. Say something brief to other players."""
        else:
            prompt = """The policy was just enacted. Say something brief about it."""

        response = self.get_basic_completion(prompt, "Discuss")

        # Remove thinking tags if present
        response = response.split("</think>")[-1].strip()

        display_player_discussion(self, response)
        return response
