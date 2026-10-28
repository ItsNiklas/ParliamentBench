import random
from typing import Optional, TYPE_CHECKING, override

from .base_player import BasePlayer
from ..game_primitives import Ja, Nein, Policy, Vote, LiberalPolicy, FascistPolicy
from ..game_logging import display_player_discussion

if TYPE_CHECKING:
    from ..game_state import GameState


class CPUPlayer(BasePlayer):
    """
    CPU-based player for Secret Hitler game with reputation-based decision making.
    """

    MAX_REPUTATION = 5

    def __init__(
        self,
        id: int,
        name: str,
        role,
        state: "GameState",
        game_log: list[str],
        chat_log: list[str],
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_tokens: int = 1024,
        request_timeout: int = 60,
        max_api_retries: int = 3,
    ) -> None:
        # Accept API credentials from game config or fall back to env vars
        super().__init__(
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

        # Tracks reputation of other players from -5 to +5
        # Lower values = more likely fascist/hitler
        self.player_reputation: dict[str, int] = {}

        # Track known roles (from game start knowledge or investigation)
        self.known_player_roles: dict[str, str] = {}

        # Track what choices we gave to chancellor in our last president turn
        self.chancellor_choices: list[str] = []

        # Track last updated round to avoid double updates
        self.last_updated_round = 1

        self.initialize_reputation()

    def initialize_reputation(self) -> None:
        """Initialize reputation and known roles based on game rules"""
        # Set all players to neutral reputation (0)
        self.player_reputation.clear()
        for player in self.state.players:
            if player.name != self.name:
                self.player_reputation[player.name] = 0

        # Update known identities based on role and game size
        self.known_player_roles.clear()

        # Add our own identity
        if self.is_fascist:
            self.known_player_roles[self.name] = "fascist"
        elif self.is_hitler:
            self.known_player_roles[self.name] = "hitler"
        else:
            self.known_player_roles[self.name] = "liberal"

        # Fascists know each other and Hitler (except in larger games)
        if self.is_fascist or (self.is_hitler and len(self.state.players) <= 6):
            for player in self.state.players:
                if player.is_fascist:
                    self.known_player_roles[player.name] = "fascist"
                elif player.is_hitler:
                    self.known_player_roles[player.name] = "hitler"

    def update_reputation(self, player_name: str, modifier: int) -> None:
        """Update a player's reputation, clamping to valid range"""
        if player_name not in self.player_reputation:
            return

        new_rep = self.player_reputation[player_name] + modifier
        new_rep = max(new_rep, -self.MAX_REPUTATION)
        new_rep = min(new_rep, self.MAX_REPUTATION)
        self.player_reputation[player_name] = new_rep

    def get_player_reputation_with_identity(self, player_name: str) -> int:
        """Get reputation, overriding with min/max if identity is known"""
        reputation = self.player_reputation.get(player_name, 0)

        if player_name in self.known_player_roles:
            role = self.known_player_roles[player_name]
            if role in ["fascist", "hitler"]:
                return -self.MAX_REPUTATION
            else:  # liberal
                return self.MAX_REPUTATION

        return reputation

    def is_fascist_in_danger(self) -> bool:
        """Check if fascists are close to losing (4+ liberal policies)"""
        return self.state.liberal_track >= 4

    def is_liberal_in_danger(self) -> bool:
        """Check if liberals are close to losing (5+ fascist policies)"""
        return self.state.fascist_track >= 5

    def can_hitler_win_by_election(self) -> bool:
        """Check if Hitler can win by being elected chancellor (3+ fascist policies)"""
        return self.state.fascist_track >= 3

    def get_eligible_players(self, exclude_self: bool = True) -> list["BasePlayer"]:
        """Get list of eligible players (alive, not self, following term limits)"""
        eligible = []

        for player in self.state.players:
            if player.is_dead:
                continue
            if exclude_self and player.name == self.name:
                continue

            # OFFICIAL RULES (F11): term limits as GameState.eligible_chancellors applies them
            if player not in self.state.eligible_chancellors(None):
                continue

            eligible.append(player)

        return eligible

    def choose_random_player_weighted(self, player_list: list["BasePlayer"], fascist_weight: float, hitler_weight: float, liberal_weight: float, user_bias: float = 0.5) -> Optional["BasePlayer"]:
        """Choose a player using weighted random based on suspected roles"""
        if not player_list:
            return None

        weights = []

        for player in player_list:
            reputation = self.get_player_reputation_with_identity(player.name)

            # Convert reputation (-5 to +5) to weight based on known/suspected role
            if player.name in self.known_player_roles:
                role = self.known_player_roles[player.name]
                if role == "hitler":
                    weight = hitler_weight
                elif role == "fascist":
                    weight = fascist_weight
                else:  # liberal
                    weight = liberal_weight
            else:
                # Interpolate based on reputation
                # reputation -5 (suspected fascist) -> fascist_weight
                # reputation +5 (suspected liberal) -> liberal_weight
                t = (reputation + self.MAX_REPUTATION) / (2 * self.MAX_REPUTATION)
                weight = fascist_weight * (1 - t) + liberal_weight * t

            # Apply user bias (assuming CPU vs human distinction isn't relevant here)
            weight *= 1 + user_bias
            weight = max(weight, 0.01)  # Minimum weight to avoid zero
            weights.append(weight)

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(player_list)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return player_list[i]

        return player_list[-1]  # Fallback

    def nominate_chancellor(self) -> "BasePlayer":
        """Nominate a chancellor based on role and game state"""
        eligible_players = self.get_eligible_players()

        if not eligible_players:
            # Should not happen, but fallback
            return random.choice([p for p in self.state.players if p != self and not p.is_dead])

        if self.is_fascist and not self.is_hitler:
            if self.can_hitler_win_by_election():
                # Try to nominate Hitler for the win
                for player in eligible_players:
                    if player.is_hitler:
                        return player
                # If Hitler not available, nominate based on reputation
                return self.choose_random_player_weighted(eligible_players, 1.0, 1.0, 0.3, 0.5)
            else:
                # Nominate fascist-friendly players
                return self.choose_random_player_weighted(eligible_players, 1.0, 0.8, 0.4, 0.5)

        elif self.is_hitler:
            # Hitler should nominate trustworthy players to avoid suspicion
            return self.choose_random_player_weighted(eligible_players, 0.3, 0.2, 0.9, 0.5)

        else:  # Liberal
            # Nominate players with good reputation
            return self.choose_random_player_weighted(eligible_players, 0.2, 0.1, 1.0, 0.5)

    def vote_government(self) -> Vote:
        """Vote on the current government based on role and reputation"""
        president_name = self.state.president.name if self.state.president else ""
        chancellor_name = self.state.chancellor.name if self.state.chancellor else ""

        president_rep = self.get_player_reputation_with_identity(president_name)
        chancellor_rep = self.get_player_reputation_with_identity(chancellor_name)

        # Modify reputation if we're Hitler and part of the government
        if self.is_hitler:
            if self.name == president_name:
                president_rep = self.MAX_REPUTATION  # Treat ourselves as liberal
            elif self.name == chancellor_name:
                chancellor_rep = self.MAX_REPUTATION

        combined_rep = president_rep + chancellor_rep

        # Normalize to [0,1] range
        t = (combined_rep + 2 * self.MAX_REPUTATION) / (4 * self.MAX_REPUTATION)

        # Special case: Fascists vote for Hitler win condition
        if self.is_fascist and self.can_hitler_win_by_election() and self.state.chancellor and self.state.chancellor.is_hitler:
            return Ja() if random.random() < 0.99 else Nein()

        # Fascist voting (including Hitler if fascists in danger)
        if (self.is_fascist and not self.is_hitler) or (self.is_hitler and self.is_fascist_in_danger()):
            fascist_vote_prob = 0.8
            liberal_vote_prob = 0.6

            if self.is_fascist_in_danger():
                fascist_vote_prob = 0.9
                liberal_vote_prob = 0.4

            vote_prob = t * liberal_vote_prob + (1 - t) * fascist_vote_prob
            return Ja() if random.random() < vote_prob else Nein()

        # Liberal/Hitler default voting
        min_individual_rep = -3
        min_combined_rep = -2

        if self.can_hitler_win_by_election():
            min_individual_rep = -2
            min_combined_rep = -1

        if president_rep < min_individual_rep or chancellor_rep < min_individual_rep or combined_rep < min_combined_rep:
            return Ja() if random.random() < 0.05 else Nein()
        else:
            # Parabolic curve: f(t) = 2t - t^2
            vote_prob = 2 * t - (t * t)
            return Ja() if random.random() < vote_prob else Nein()

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        """Filter policies as president (discard one of three)"""
        fascist_count = sum(1 for p in policies if isinstance(p, FascistPolicy))
        liberal_count = len(policies) - fascist_count

        # If all same type, discard any
        if fascist_count == len(policies) or liberal_count == len(policies):
            discard_index = 0
        elif fascist_count == 1:
            # One fascist policy - decision based on role
            if self.is_fascist and not self.is_hitler:
                # Regular fascists keep the fascist policy, discard a liberal
                discard_index = next(i for i, p in enumerate(policies) if isinstance(p, LiberalPolicy))
            else:  # Liberal or Hitler (Hitler acts like liberal)
                # Discard the fascist policy
                discard_index = next(i for i, p in enumerate(policies) if isinstance(p, FascistPolicy))
        else:  # fascist_count == 2
            if self.is_fascist and not self.is_hitler:
                # Regular fascists keep fascist policies, discard liberal
                discard_index = next(i for i, p in enumerate(policies) if isinstance(p, LiberalPolicy))
            else:  # Liberal or Hitler (Hitler acts like liberal)
                # Discard a fascist policy
                discard_index = next(i for i, p in enumerate(policies) if isinstance(p, FascistPolicy))

        # Track what we're giving to chancellor
        remaining_policies = [p for i, p in enumerate(policies) if i != discard_index]
        self.chancellor_choices = [type(p).__name__ for p in remaining_policies]

        discarded = policies[discard_index]
        return remaining_policies, discarded

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """Enact one of two policies as chancellor"""
        if len(policies) != 2:
            raise ValueError("Chancellor should receive exactly 2 policies")

        policy1, policy2 = policies

        # If both same type, doesn't matter which
        if type(policy1) == type(policy2):
            chosen = policy1
            discarded = policy2
        else:
            # One liberal, one fascist - choose based on role
            if self.is_fascist and not self.is_hitler:
                # Regular fascists choose fascist policy
                if isinstance(policy1, FascistPolicy):
                    chosen, discarded = policy1, policy2
                else:
                    chosen, discarded = policy2, policy1
            else:  # Liberal or Hitler (Hitler acts like liberal)
                # Choose liberal policy
                if isinstance(policy1, LiberalPolicy):
                    chosen, discarded = policy1, policy2
                else:
                    chosen, discarded = policy2, policy1

        return chosen, discarded

    def veto(self, policies: list[Policy]) -> bool:
        """Decide whether to veto as chancellor (only available with 5 fascist policies)"""
        if self.state.fascist_track < 5:
            return False

        # If both policies same type
        if type(policies[0]) == type(policies[1]):
            policy_type = type(policies[0])
            if self.role.party_membership == "liberal":
                # Liberals veto if both fascist
                return isinstance(policies[0], FascistPolicy)
            elif self.is_hitler:
                # Hitler acts like liberal - veto if both fascist
                return isinstance(policies[0], FascistPolicy)
            else:  # Regular fascist
                # Fascists veto if both liberal (to delay liberal win)
                return isinstance(policies[0], LiberalPolicy)
        else:
            # One of each - no reason to veto since we have choice
            return False

    def view_policies(self, policies: list[Policy]) -> None:
        """Handle viewing top 3 policies (peek power) - currently no special action"""
        pass

    def inspect_player(self) -> "BasePlayer":
        """Choose a player to investigate their party membership"""
        eligible = self.state.eligible_investigation_targets(self)  # OFFICIAL RULES (F13)

        if self.role.party_membership == "liberal" or self.is_hitler:
            # Liberals and Hitler investigate suspicious players
            return self.choose_random_player_weighted(eligible, 1.0, 1.0, 0.2, 0.5)
        else:  # Regular fascist
            # Fascists investigate to gain info or mislead
            return self.choose_random_player_weighted(eligible, 0.4, 0.3, 1.0, 0.5)

    def kill(self) -> "BasePlayer":
        """Choose a player to execute"""
        eligible = [p for p in self.state.players if p.name != self.name and not p.is_dead]

        if self.role.party_membership == "liberal" or self.is_hitler:
            # Liberals and Hitler kill suspicious players
            return self.choose_random_player_weighted(eligible, 1.0, 1.0, 0.1, 0.5)
        else:  # Regular fascist
            # Fascists kill liberals
            return self.choose_random_player_weighted(eligible, 0.1, 0.1, 1.0, 0.5)

    def choose_next(self) -> "BasePlayer":
        """Choose next president (special election power)"""
        eligible = [p for p in self.state.players if p.name != self.name and not p.is_dead]

        if self.role.party_membership == "liberal" or self.is_hitler:
            # Liberals and Hitler choose trustworthy players
            return self.choose_random_player_weighted(eligible, -0.8, -0.8, 1.0, 0.5)
        else:  # Regular fascist
            # Fascists choose fascist-friendly players but not too obviously
            return self.choose_random_player_weighted(eligible, 1.0, 1.0, 0.5, 0.5)

    def update_after_legislation(self) -> None:
        """Update reputation after legislation passes (called each round)"""
        current_round = getattr(self.state, "turn_count", 1)

        if current_round <= self.last_updated_round:
            return  # Already updated this round

        if not self.state.most_recent_policy:
            return  # No policy enacted yet

        last_policy_type = type(self.state.most_recent_policy).__name__

        # Don't update reputation if election tracker advanced (failed votes)
        if self.state.failed_votes > 0:
            self.last_updated_round = current_round
            return

        president_name = self.state.president.name if self.state.president else ""
        chancellor_name = self.state.chancellor.name if self.state.chancellor else ""

        # Skip if we were president (we know our own choices)
        if president_name == self.name:
            self.last_updated_round = current_round
            return

        # Update reputation based on what was enacted
        if last_policy_type == "LiberalPolicy":
            # Liberal policy enacted - increase reputation of both players
            if president_name:
                self.update_reputation(president_name, 1)
            if chancellor_name:
                self.update_reputation(chancellor_name, 1)
        else:  # FascistPolicy
            # Fascist policy enacted - decrease reputation
            if president_name:
                self.update_reputation(president_name, -1)
            if chancellor_name:
                self.update_reputation(chancellor_name, -2)  # Chancellor had final choice

        self.last_updated_round = current_round

    def update_known_role_from_investigation(self, player_name: str, revealed_party: str) -> None:
        """Update known roles when a player is investigated"""
        # Convert party membership to role (investigation only reveals party, not specific role)
        if revealed_party == "fascist":
            # Could be fascist or hitler, but we mark as fascist for now
            self.known_player_roles[player_name] = "fascist"
        else:  # liberal
            self.known_player_roles[player_name] = "liberal"

    @override
    def discuss(self, chat: str, stage: str) -> str:
        top_suspects = sorted(
            self.player_reputation.items(),
            key=lambda kv: kv[1]
        )
        suspect_text = top_suspects[0][0] if top_suspects else "none"
        if stage == "discussion_on_potential_government":
            response = (
                f"I currently distrust {suspect_text} the most. "
                "My vote follows reputation and policy pressure."
            )
        elif stage == "after_policy":
            response = (
                f"Recent policy outcomes increased my suspicion of {suspect_text}. "
                "We should compare this with voting behavior."
            )
        else:
            response = "I am updating reputations from the latest round."

        # Use the appropriate function to display player discussion
        display_player_discussion(self, response)

        # Log timestamp with proper formatting
        # logger.debug("----------------------------")
        # logger.debug(f"DISCUSSION PHASE: {stage}")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

        chat += f"{str(self)}: {response}\n\n"
        return response

    @override
    def reflect_on_roles(self) -> str:
        return ""

    @override
    def rapid_role_assessment(self) -> str:
        other_players = [player for player in self.state.players if player != self and not player.is_dead]
        if not other_players:
            return ""

        assessments: list[str] = []
        for player in other_players:
            if player.name in self.known_player_roles:
                known_role = self.known_player_roles[player.name]
                if self.role.party_membership == "liberal":
                    if known_role == "hitler":
                        assessments.append(f"{player.name}: Hitler")
                    elif known_role == "fascist":
                        assessments.append(f"{player.name}: Fascist")
                    else:
                        assessments.append(f"{player.name}: Liberal")
                else:
                    if known_role in ["fascist", "hitler"]:
                        assessments.append(f"{player.name}: Liberal")
                    else:
                        assessments.append(f"{player.name}: Liberal")
            else:
                rep = self.get_player_reputation_with_identity(player.name)
                if rep <= -3:
                    assessments.append(f"{player.name}: Fascist")
                elif rep <= -1:
                    assessments.append(f"{player.name}: Unknown")
                elif rep >= 2:
                    assessments.append(f"{player.name}: Liberal")
                else:
                    assessments.append(f"{player.name}: Unknown")

        return "\n".join(assessments)
