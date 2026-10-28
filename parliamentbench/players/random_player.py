from random import choice, getrandbits
import random

from .base_player import BasePlayer
from ..game_primitives import Ja, Nein, Policy, Vote
from ..game_logging import display_player_discussion


class RandomPlayer(BasePlayer):
    def vote_government(self) -> Vote:
        return random.choice([Ja(), Nein()])
    
    def nominate_chancellor(self) -> "BasePlayer":
        eligible_players = self.state.eligible_chancellors(self)  # OFFICIAL RULES (F11)
        
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        if len(policies) != 3:
            return (policies[:-1], policies[-1])
        discard_index = random.randint(0, 2)
        discarded = policies[discard_index]
        remaining = [policies[i] for i in range(3) if i != discard_index]
        return (remaining, discarded)
    
    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        if len(policies) != 2:
            return (policies[0], policies[1] if len(policies) > 1 else policies[0])
        if random.choice([True, False]):
            return (policies[0], policies[1])
        else:
            return (policies[1], policies[0])
    
    def veto(self, policies: list[Policy]) -> bool:
        return bool(getrandbits(1))
    
    def view_policies(self, policies: list[Policy]) -> None:
        pass
    
    def kill(self) -> "BasePlayer":
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def inspect_player(self) -> "BasePlayer":
        eligible_players = self.state.eligible_investigation_targets(self)  # OFFICIAL RULES (F13)
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def choose_next(self) -> "BasePlayer":
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        if not eligible_players:
            return choice(self.state.players)
        return choice(eligible_players)
    
    def discuss(self, chat: str, stage: str) -> str:
        canned = [
            "I am not fully convinced yet.",
            "Let's keep an eye on voting patterns.",
            "This government could work.",
            "I want more evidence before strong claims.",
            "We should reconsider after the next round.",
        ]
        response = random.choice(canned)

        # Use the appropriate function to display player discussion
        display_player_discussion(self, response)

        chat += f"{str(self)}: {response}\n\n"
        return response

    def reflect_on_roles(self) -> str:
        # Random player has no meaningful reflection
        return "I'm just going with my gut feeling here."

    def rapid_role_assessment(self) -> str:
        other_players = [player for player in self.state.players if player != self and not player.is_dead]
        if not other_players:
            return ""
        options = ["Liberal", "Fascist", "Hitler", "Unknown"]
        return "\n".join(f"{player.name}: {random.choice(options)}" for player in other_players)
