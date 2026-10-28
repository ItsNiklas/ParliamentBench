from random import choice, getrandbits
import random

from .base_player import BasePlayer
from ..game_primitives import Ja, Nein, Policy, Vote
from ..game_logging import (
    logger,
    display_player_reasoning,
    display_policy_table,
    display_player_discussion,
)


class LLMPlayer(BasePlayer):
    def __init__(self, id, name: str, role, state, game_log, chat_log, player_index: int = 0, api_key: str = None, base_url: str = None, model: str = None, max_tokens: int = 1024, request_timeout: int = 60, max_api_retries: int = 3) -> None:
        # Accept API credentials from game config or fall back to env vars
        super(LLMPlayer, self).__init__(
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

    def vote_government(self) -> Vote:
        """
        :return: Ja or Nein
        """

        prompt = f"""
        It is now your turn to vote for the nominated chancellor.

        It is important to note that your vote is public and will be revealed to the other players.

        Here is the state of the board, where "president" indicates the proposed president and "chancellor" indicates the nominated chancellor.

        {self.get_known_state()}

        "JA" means yes, and "NEIN" means no.
        You will first explain your inner thoughts and reasoning (which are private to you), then you will vote ONLY either with "FINAL VOTE: JA" or "FINAL VOTE: NEIN" at the very end.
        """

        response = self.get_completion(prompt, "Vote")

        display_player_reasoning(
            self, response, f"{self.name}'s Voting Thoughts", "blue"
        )

        # logger.debug("----------------------------")
        # logger.debug("VOTING")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

        if "FINAL VOTE: JA" in response.upper():
            return Ja()
        elif "FINAL VOTE: NEIN" in response.upper():
            return Nein()

        logger.debug("No vote selected, returning random vote.")
        return random.choice([Ja(), Nein()])

    def nominate_chancellor(self) -> "BasePlayer":
        """
        More random!
        :return: BasePlayer
        one of self.state.players
        """
        eligible_players = self.state.eligible_chancellors(self)  # OFFICIAL RULES (F11)

        prompt = f"""
        It is now your turn to nominate a chancellor.

        Here is the state of the board:

        {self.get_known_state()}

        VALID OPTIONS (you MUST choose one of these):
            {["FINAL SELECTION: " + player.name for player in eligible_players]}
        
        Do NOT nominate yourself, the last elected chancellor, dead players, or (while more than five players are alive) the last elected president.

        You will first explain your inner thoughts and reasoning (which are private to you), then you will nominate EXACTLY and ONLY with one of the VALID OPTIONS above.
        """

        response = self.get_completion(prompt, "Nominate Chancellor")

        display_player_reasoning(
            self, response, f"{self.name}'s Chancellor Nomination", "cyan"
        )

        # Parse response - try both exact match and partial match
        for player in eligible_players:
            if f"FINAL SELECTION: {player.name.upper()}" in response.upper():
                return player
        
        # Fallback: check if player name appears anywhere in response
        for player in eligible_players:
            if player.name.upper() in response.upper():
                logger.debug(f"Found player name {player.name} in response (partial match)")
                return player

        logger.debug("No player nominated, returning random eligible player.")
        return choice(eligible_players) if eligible_players else choice(self.state.players)

    def view_policies(self, policies: list[Policy]) -> None:
        """
        What to do if you perform the presidential action to view the top three policies
        :return:
        """
        prompt = f"""
        It is now your turn to view the top three policies in the policy deck.
        
        Here is the state of the board:
        
        {self.get_known_state()}
        
        The policies you see are:
        Policy 1: {str(policies[0])}
        Policy 2: {str(policies[1])}
        Policy 3: {str(policies[2])}
        
        Please describe your inner thoughts about what you are seeing. This information is for your eyes only and will help inform your future decisions.
        """

        response = self.get_completion(prompt, "View Policies")

        # Display the policies in a nice table format
        display_policy_table(policies, f"Policies viewed by {self.name}", True)

        # Display the player's reasoning
        display_player_reasoning(
            self, response, f"{self.name}'s Policy Analysis", "yellow"
        )

        # logger.debug("----------------------------")
        # logger.debug("VIEWING POLICIES")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

    def kill(self) -> "BasePlayer":
        """
        Choose a person to kill
        :return: BasePlayer
        """

        # Get eligible players (alive and not self)
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        
        if not eligible_players:
            logger.error("No eligible players to execute!")
            return choice(self.state.players)

        prompt = f"""
        It is now your turn to execute a player using your presidential power.
        
        Here is the state of the board:

        {self.get_known_state()}

        VALID OPTIONS (you MUST choose one of these):
            {["FINAL EXECUTION: " + player.name for player in eligible_players]}
        
        You may pick any living player EXCEPT yourself.

        You will first explain your inner thoughts and reasoning (which are private to you), then you will execute and respond EXACTLY and ONLY with one of the VALID OPTIONS above.
        """

        response = self.get_completion(prompt, "Kill a Player")

        # Display the player's reasoning using the proper function
        display_player_reasoning(
            self, response, f"{self.name}'s Execution Decision", "red"
        )
        self.inspection += f"\n\n[Kill Decision]\n{response}\n"

        # Parse the response to find the selected player
        for player in eligible_players:
            if f"FINAL EXECUTION: {player.name.upper()}" in response.upper():
                return player
        
        # Fallback: check if player name appears anywhere
        for player in eligible_players:
            if player.name.upper() in response.upper():
                logger.debug(f"Found player name {player.name} in response (partial match)")
                return player

        logger.debug("No player selected for execution, returning random eligible player.")
        return choice(eligible_players)

    def inspect_player(self) -> "BasePlayer":
        """
        Choose a person's party membership to inspect
        :return:
        """

        eligible_players = self.state.eligible_investigation_targets(self)  # OFFICIAL RULES (F13)
        
        if not eligible_players:
            logger.error("No eligible players to inspect!")
            return choice(self.state.players)

        prompt = f"""It is now your turn. You must use your executive power to inspect someone's party membership. You should note that the player you inspect and their party membership will be revealed to you and only you. It is up to you to share this information with the other players.

        The current state of the game is as follows:

        {self.get_known_state()}

        VALID OPTIONS (you MUST choose one of these):
            {["FINAL INVESTIGATION: " + player.name for player in eligible_players]}
        
        You should NOT inspect yourself (you already know your role), dead players, or anyone who has already been investigated.

        First, please describe what your inner thoughts and strategies are for this current move (they are private to you). Your future self will reference this strategy on the next turn when deciding what to do.
        Then you will investigate and respond EXACTLY and ONLY with one of the VALID OPTIONS above.
        """

        response = self.get_completion(prompt, "Inspect Player")

        # Display the player's reasoning
        display_player_reasoning(
            self, response, f"{self.name}'s Inspection Decision", "yellow"
        )
        self.inspection += f"\n\n[Inspection Decision]\n{response}\n"

        # Parse the response to find the selected player
        for player in eligible_players:
            if f"FINAL INVESTIGATION: {player.name.upper()}" in response.upper():
                return player
        
        # Fallback: check if player name appears anywhere
        for player in eligible_players:
            if player.name.upper() in response.upper():
                logger.debug(f"Found player name {player.name} in response (partial match)")
                return player

        logger.debug("No player selected for inspection, returning random eligible player.")
        return choice(eligible_players)

    def choose_next(self) -> "BasePlayer":
        """
        Choose the next president
        :return:
        """
        
        # Get eligible players (alive and not self)
        eligible_players = [player for player in self.state.players 
                          if player != self and not player.is_dead]
        
        if not eligible_players:
            logger.error("No eligible players to choose as next president!")
            return choice(self.state.players)

        prompt = f"""It is now your turn. You must use your executive power to choose the next president. The current state of the game is as follows:

        {self.get_known_state()}

        VALID OPTIONS (you MUST choose one of these):
            {["FINAL CHOICE: " + player.name for player in eligible_players]}
        
        You must NOT choose yourself or dead players.

        First, please describe what your inner thoughts and strategy are for this current move (they are private to you). Your future self will reference this strategy on the next turn when deciding what to do. Consider this like a monologue.
        Then you will choose the next president and respond EXACTLY and ONLY with one of the VALID OPTIONS above.
        """

        response = self.get_completion(prompt, "Choose the next president")

        # Display the player's reasoning
        display_player_reasoning(
            self, response, f"{self.name}'s Presidential Selection", "cyan"
        )
        self.inspection += f"\n\n[Presidential Selection]\n{response}\n"

        # Parse the response to find the selected player
        for player in eligible_players:
            if f"FINAL CHOICE: {player.name.upper()}" in response.upper():
                return player
        
        # Fallback: check if player name appears anywhere
        for player in eligible_players:
            if player.name.upper() in response.upper():
                logger.debug(f"Found player name {player.name} in response (partial match)")
                return player

        logger.debug("No player selected for next president, returning random eligible player.")
        return choice(eligible_players)

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        prompt = f"""
        It is your turn:

        {self.get_known_state()}

        This is part of the governing process. The president and chancellor are working together to enact a policy. After this, a new government will be formed.

        You are the chancellor. You have been given two cards, and you must choose one to discard, and one to enact. This is secret information, and nobody else will know which card you discarded.
        These are the two cards you have been given by the president:
        
        Card 1: {str(policies[0])},
        Card 2: {str(policies[1])}.

        You must pick one to discard, and one to enact as policy.
        First, please describe what your inner thoughts and strategy are (they are private to you). Your future self will reference this strategy on the next turn when deciding what to do. Consider this like a monologue.
        Then, choose the policy by saying "DISCARD: Card 1" or "DISCARD: Card 2".
        """
        # Display the available policies as a table
        display_policy_table(
            policies, f"Chancellor {self.name} selects from two policies", True
        )

        response = self.get_completion(prompt, "Enact a policy (Chancellor)")

        # Display the player's reasoning
        display_player_reasoning(
            self, response, f"{self.name}'s Policy Decision", "blue"
        )

        # logger.debug("----------------------------")
        # logger.debug("ENACTING POLICY")
        # logger.debug(f"Chancellor: {self.name}")
        # logger.debug("----------------------------")

        if "DISCARD: CARD 1" in response.upper():
            logger.debug(f"Discarding card 1: {policies[0]}")
            return (policies[1], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            logger.debug(f"Discarding card 2: {policies[1]}")
            return (policies[0], policies[1])

        logger.debug("No policy selected, returning random policy.")
        return (policies[0], policies[1])

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        prompt = f"""
        It is your turn:

        {self.get_known_state()}

        This is part of the governing process. The president and chancellor are working together to enact a policy. After this, a new government will be formed.
        
        As president, you have drawn 3 cards secretly. These will correspond to policies that will be enacted. You can secretly discard one card and give the remaining two to the chancellor, and the chancellor will choose one of the two cards to enact. Nobody will know which card you discarded. Only the chancellor will know which cards you passed on:
        
        Card 1: {str(policies[0])},
        Card 2: {str(policies[1])},
        Card 3: {str(policies[2])}.

        You will first explain your inner thoughts and reasoning (they are private to you), then you will respond to this with the card you choose to DISCARD.

        Choose the card to discard from one of the following options VERBATIM:
            DISCARD: Card 1
            DISCARD: Card 2
            DISCARD: Card 3
        """
        # Display the policies as a nicely formatted table
        display_policy_table(
            policies, f"President {self.name} draws three policies", True
        )

        response = self.get_completion(prompt, "Choose a policy to discard (President)")

        # Display the player's reasoning
        display_player_reasoning(
            self, response, f"{self.name}'s Policy Selection", "blue"
        )

        # logger.debug("----------------------------")
        # logger.debug("FILTERING POLICY")
        # logger.debug(f"President: {self.name}")
        # logger.debug("----------------------------")

        if "DISCARD: CARD 1" in response.upper():
            logger.debug(f"Discarding card 1: {policies[0]}")
            return ([policies[1], policies[2]], policies[0])
        elif "DISCARD: CARD 2" in response.upper():
            logger.debug(f"Discarding card 2: {policies[1]}")
            return ([policies[0], policies[2]], policies[1])
        elif "DISCARD: CARD 3" in response.upper():
            logger.debug(f"Discarding card 3: {policies[2]}")
            return ([policies[0], policies[1]], policies[2])

        logger.debug("No policy selected, returning random policy.")
        return ([policies[0], policies[1]], policies[2])

    def veto(self, policies: list[Policy]) -> bool:
        """
        Decide whether to veto an action or not
        :param policies: The policies currently being considered for veto
        :return: Boolean
        """
        prompt = f"""
        It is now your turn to decide whether to veto the current policy options.
        
        The policies you are considering for veto are: {[p.type for p in policies]}
        
        Here is the state of the board:
        
        {self.get_known_state()}
        
        Vetoing means rejecting all available policy options. This counts as a failed government and advances the election tracker.
        Both the President and Chancellor must agree to veto for it to take effect.
        
        You will first explain your inner thoughts and reasoning (which are private to you), then respond with ONLY "FINAL DECISION: VETO" or "FINAL DECISION: NO VETO".
        """

        response = self.get_completion(prompt, "Veto Decision")

        # Display the player's reasoning about veto
        display_player_reasoning(
            self, response, f"{self.name}'s Veto Consideration", "yellow"
        )

        # logger.debug("----------------------------")
        # logger.debug("VETO DECISION")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

        decision = None
        if "FINAL DECISION: VETO" in response.upper():
            decision = True
        elif "FINAL DECISION: NO VETO" in response.upper():
            decision = False
        else:
            # Fall back to random if no clear decision
            decision = bool(getrandbits(1))
            logger.debug(f"No clear veto decision, randomly choosing: {decision}")

        return decision

    def discuss(self, chat: str, stage: str) -> str:
        prompt = f"""
        It is your turn:

        {self.get_known_state()}

        It is currently time to discuss. The current stage is {stage}. This refers to whether you are discussing the policy that was just enacted, or if you are discussing whether to vote on a government.

        You MUST DIRECTLY RESPOND with what you are saying to the rest of the players.
        """

        if stage == "discussion_on_potential_government":
            prompt += """
            Your goal is to convince the other players to vote either JA (yes) or NEIN (no), depending on what your strategy is.
            However, you should not actually reveal what your strategy is. You should only try to convince the other players to vote in a certain way. Please keep your response brief and to the point.

            Now, respond to the other players once. All players will review the responses before choosing their vote.
            Your goal is to convince other players to make the decision that benefits you and your team. BE PERSUASIVE WHEN NECESSARY, BUT KEEP RESPONSES SUCCINCT.

            If you feel that you have specific information that will be good for anyone to know, then tell them. You may lie!
            """
        elif stage == "after_policy":
            prompt += """
            Your goal is to analyze the policy that was just enacted, and to see whether any of the players in the government are suspicious. You should especially be looking for any inconsistencies in the story that the president and chancellor are telling.
            If a fascist policy was enacted, take this into account when analyzing the situation. Remember, YOUR ULTIMATE GOAL IS ADVANCING YOUR TEAM'S AGENDA.

            Now, respond to the other players. If you have any new information (for example, if you have insight into the previous voting round as a president or chancellor), then consider sharing this information. If you had previously inspected a player and your "known_fascists" list has changed, then consider sharing this information.
            
            Still, be concise."""

        response = self.get_completion(prompt, "Discuss with other players")

        # Remove Qwen reasoning model's thinking tags from discussion responses
        # Always split on </think> and take everything after it (the actual response)
        response = response.split("</think>")[-1].strip()

        # Use the appropriate function to display player discussion
        display_player_discussion(self, response)

        # Log timestamp with proper formatting
        # logger.debug("----------------------------")
        # logger.debug(f"DISCUSSION PHASE: {stage}")
        # logger.debug(f"Player: {self.name}")
        # logger.debug("----------------------------")

        chat += f"{str(self)}: {response}\n\n"
        return response
