from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .base_player import BasePlayer
from ..game_primitives import Ja, Nein, Policy, Vote
from ..game_logging import display_game_status, display_info_message

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..game_state import GameState


class HumanPlayer(BasePlayer):
    """CLI-based player that forwards every game decision to a human user."""

    def __init__(
        self,
        id: int,
        name: str,
        role,
        state: "GameState",
        game_log: list[str],
        chat_log: list[str],
        max_tokens: int = 1024,
        request_timeout: int = 60,
        max_api_retries: int = 3,
    ) -> None:
        super().__init__(
            id,
            name,
            role,
            state,
            game_log,
            chat_log,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            max_api_retries=max_api_retries,
        )
        # Human interaction requires synchronous prompts.
        BasePlayer.enable_parallel_processing = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _print_header(self, action: str) -> None:
        banner = f"\n[Human Player – {self.name}] {action}\n" + "-" * 60
        print(banner, flush=True)
        display_game_status(self.state)

    def _prompt(self, prompt: str, allow_empty: bool = False, default: str | None = None) -> str:
        while True:
            try:
                response = input(f"{prompt}: ").strip()
            except EOFError:
                return default or ""
            except KeyboardInterrupt:
                print("", file=sys.stderr)
                raise

            if response:
                return response
            if allow_empty:
                return default or ""
            if default is not None:
                return default
            print("Please enter a value.", flush=True)

    def _prompt_choice(self, label: str, options: list[str]) -> int:
        if not options:
            raise ValueError("No options available for prompt")

        while True:
            print(f"{label}:", flush=True)
            for idx, option in enumerate(options, start=1):
                print(f"  {idx}) {option}", flush=True)

            response = self._prompt("Select an option by number")

            if response.isdigit():
                index = int(response)
                if 1 <= index <= len(options):
                    return index - 1

            normalized = response.lower()
            for idx, option in enumerate(options):
                if normalized == option.lower():
                    return idx

            print("Invalid selection. Please try again.", flush=True)

    def _format_policies(self, policies: list[Policy]) -> list[str]:
        return [f"{i + 1}) {policy}" for i, policy in enumerate(policies)]

    # ------------------------------------------------------------------
    # Core gameplay decisions
    # ------------------------------------------------------------------
    def vote_government(self) -> Vote:
        self._print_header("Vote for Government")

        president = self.state.president.name if self.state.president else "Unknown"
        chancellor = self.state.chancellor.name if self.state.chancellor else "Unknown"
        print(f"Proposed president: {president}", flush=True)
        print(f"Proposed chancellor: {chancellor}", flush=True)

        choice_index = self._prompt_choice("Cast your vote", ["JA (approve)", "NEIN (reject)"])
        return Ja() if choice_index == 0 else Nein()

    def nominate_chancellor(self) -> "BasePlayer":
        self._print_header("Nominate a Chancellor")

        eligible = self.state.eligible_chancellors(self)  # OFFICIAL RULES (F11)

        if not eligible:
            display_info_message("No eligible players found; selecting randomly.")
            return self.state.players[0]

        options = [player.name for player in eligible]
        index = self._prompt_choice("Choose your nominated chancellor", options)
        return eligible[index]

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        self._print_header("Presidential Policy Choice")
        option_strings = self._format_policies(policies)
        discard_idx = self._prompt_choice("Select a policy to discard", option_strings)
        discarded = policies[discard_idx]
        remaining = [policy for i, policy in enumerate(policies) if i != discard_idx]
        return remaining, discarded

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        if len(policies) != 2:
            raise ValueError("Chancellor must receive exactly two policies")

        self._print_header("Chancellor Policy Choice")
        option_strings = self._format_policies(policies)
        enact_idx = self._prompt_choice("Select a policy to enact", option_strings)
        chosen = policies[enact_idx]
        discarded = policies[1 - enact_idx]
        return chosen, discarded

    def veto(self, policies: list[Policy]) -> bool:
        self._print_header("Veto Decision")
        for line in self._format_policies(policies):
            print(line, flush=True)
        choice_index = self._prompt_choice("Would you like to veto these policies?", ["Yes", "No"])
        return choice_index == 0

    def view_policies(self, policies: list[Policy]) -> None:
        self._print_header("Policy Peek")
        for line in self._format_policies(policies):
            print(line, flush=True)
        self._prompt("Press Enter after acknowledging the policies", allow_empty=True)

    def kill(self) -> "BasePlayer":
        self._print_header("Execution Decision")
        eligible = [player for player in self.state.players if player != self and not player.is_dead]
        if not eligible:
            raise ValueError("No executives options available")
        options = [player.name for player in eligible]
        index = self._prompt_choice("Choose a player to execute", options)
        return eligible[index]

    def inspect_player(self) -> "BasePlayer":
        self._print_header("Investigation Decision")
        eligible = self.state.eligible_investigation_targets(self)  # OFFICIAL RULES (F13)
        if not eligible:
            raise ValueError("No inspection targets available")
        options = [player.name for player in eligible]
        index = self._prompt_choice("Choose a player to investigate", options)
        return eligible[index]

    def choose_next(self) -> "BasePlayer":
        self._print_header("Special Election Decision")
        eligible = [player for player in self.state.players if player != self and not player.is_dead]
        if not eligible:
            raise ValueError("No candidates available for special election")
        options = [player.name for player in eligible]
        index = self._prompt_choice("Choose the next president", options)
        return eligible[index]

    def discuss(self, chat: str, stage: str) -> str:
        self._print_header(f"Discussion Stage: {stage}")
        if chat:
            print(chat, flush=True)
        message = self._prompt("Enter your discussion message (blank to pass)", allow_empty=True)
        return message or "[No comment]"

    # ------------------------------------------------------------------
    # Optional reflection helpers
    # ------------------------------------------------------------------
    def reflect_on_roles(self) -> str:
        self._print_header("Private Reflection")
        print("Provide a short monologue about who might be Hitler or a fascist.", flush=True)

        reflection_lines: list[str] = []
        while True:
            line = self._prompt("Enter reflection line (blank to finish)", allow_empty=True)
            if not line:
                break
            reflection_lines.append(line)

        reflection_text = "\n".join(reflection_lines) if reflection_lines else "(No reflection recorded.)"

        categories = {
            "A": "Recent policy (laws passed, voting outcomes)",
            "B": "Probability-based reasoning (stats, patterns)",
            "C": "Statements made by other players",
            "D": "Random guess / intuition",
            "NONE": "Doesn't fit — propose a new category",
        }

        print("Select the primary reasoning category:", flush=True)
        for key, description in categories.items():
            print(f"  {key}: {description}", flush=True)

        while True:
            category_input = self._prompt("Enter category (A/B/C/D/NONE or custom)").strip()
            if not category_input:
                continue

            upper = category_input.upper()
            if upper in categories:
                category_value = upper
                break

            category_value = category_input
            break

        result = f"{reflection_text}\nReasoning Category: {category_value}"
        self.inspection += f"\n\n[Reflection after vote]\n{result}\n"
        return result

    def rapid_role_assessment(self) -> str:
        self._print_header("Rapid Role Assessment")
        other_players = [player for player in self.state.players if player != self and not player.is_dead]

        if not other_players:
            return ""

        options = ["Liberal", "Fascist", "Hitler", "Unknown"]
        lines: list[str] = []

        for player in other_players:
            prompt = f"Assess {player.name}" + " (choose option)"
            choice_index = self._prompt_choice(prompt, options)
            role_label = options[choice_index]
            lines.append(f"{player.name}: {role_label}")

        assessment_text = "\n".join(lines)
        self.inspection += f"\n\n[Rapid Role Assessment]\n{assessment_text}\n"
        return assessment_text
