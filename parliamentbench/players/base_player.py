from typing import TYPE_CHECKING

from openai import OpenAI

from ..game_primitives import Policy, Role, Vote
from ..game_logging import *
from ..metric.token_tracker import track_response
from ..theme import apply_to_messages

if TYPE_CHECKING:
    from ..game_state import GameState

# from pinecone import Pinecone, ServerlessSpec

# pinecone_api_key = os.getenv('PINECONE_API_KEY')
# pc = Pinecone(api_key=pinecone_api_key)

# index_name = 'secret-hitler-strategy'
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=3072, # Replace with your model dimensions
#         metric="cosine", # Replace with your model metric
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# pinecone_index = pc.Index(index_name)
# pinecone_index = pc.Index(index_name)


class BasePlayer:
    # Configuration for batching behavior (class-level setting)
    enable_parallel_processing = True

    # Generation knobs (class-level). Resolved from config at startup by the
    # benchmark runner; the defaults reproduce the original behaviour.
    reasoning_enabled = True          # False => thinking-OFF bundle
    reasoning_effort = "low"          # reasoning effort when enabled: low|high
    max_retries = 3                   # completion attempts before giving up
    completion_max_tokens = 1024 * 4  # LLMPlayer base; scales x(attempt+1)
    basic_max_tokens = 512            # BasicLLMPlayer base; scales x(attempt+1)

    def __init__(
        self,
        id: int,
        name: str,
        role: Role,
        state: "GameState",
        game_log: list[str],
        chat_log: list[str],
        api_key: str = "",
        base_url: str = "http://localhost:8080/v1/",
        model: str = None,
        max_tokens: int = 1024,
        request_timeout: int = 60,
        max_api_retries: int = 3,
    ) -> None:
        self.id = id
        self.name = name
        self.role = role
        self.state = state
        self.fascists: list[BasePlayer] = []
        self.hitler = None
        self.is_dead = False
        self.inspected_players = ""
        self.inspection = ""
        
        # Instance-level LLM client configuration
        self.openai_api_key = api_key
        self.openai_base_url = base_url
        self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
        self.request_timeout = request_timeout
        # `max_tokens` / `max_api_retries` are retained for API compatibility.
        # The budgets and retry count actually used by the completion path come
        # from the class-level generation knobs above, which the runner resolves
        # once at startup via `Config.effective_generation()` — that resolution
        # already falls back to these game-level values when the config has no
        # `generation:` section.
        self.max_tokens = max_tokens
        self.max_api_retries = max_api_retries
        
        # Cache the model name for this player (use override if provided)
        self._model_name = model

    def reflect_on_roles(self) -> str:
        prompt = f"""Write a short monologue expressing your opinion about who you think might be Hitler and who you believe could be a fascist, based on your knowledge so far.

        Current game state:
        {self.get_known_state()}

        Then, clearly select ONE of the following categories that best describes the main basis for your deduction. Use the **exact format** shown: `"Reasoning Category: X"`, where X is one of the letters or a new category if needed.

        A: Recent policy (e.g., laws passed, voting outcomes)
        
        B: Probability-based reasoning (e.g., statistical likelihood, pattern recognition)
        
        C: Statements made by other players
        
        D: Random guess / intuition
        
        NONE: Doesn't fit — propose a new category"""

        response = self.get_completion(prompt, _stage="reflection")
        self.inspection += f"\n\n[Reflection after vote]\n{response}\n"
        return response

    def rapid_role_assessment(self) -> str:
        """Quick assessment of each other player's suspected role"""
        other_players = [player for player in self.state.players if player != self and not player.is_dead]

        if not other_players:
            return ""

        # Build the format with only existing players
        format_lines = []
        for player in other_players:
            format_lines.append(f"{player.name}: [Liberal/Fascist/Hitler/Unknown]")

        prompt = f"""Give an assessment of each other player's role based on the game state and chat messages. Be very brief and direct.

        Respond in this exact format for each player:
        {"\n".join(format_lines)}
        
        Choose one option for each: Liberal, Fascist, Hitler, or Unknown.
        No explanations needed - just the quick assessment."""

        response = self.get_completion(prompt, _stage="rapid_assessment")
        self.inspection += f"\n\n[Rapid Role Assessment]\n{response}\n"
        return response

    def get_model_name(self) -> str:
        """Get the model name for this player, caching it after first retrieval.
        
        If a model override was provided via config, that is used directly.
        Otherwise, auto-detects by querying the API's models endpoint.
        """
        if self._model_name is None:
            try:
                self._model_name = self.openai_client.models.list().data[0].id
            except Exception as e:
                logger.warning(f"Failed to get model name for {self.name}: {e}")
                self._model_name = "unknown"
        return self._model_name

    def get_completion(self, prompt: str, _stage: str) -> str:
        openai_model = self.get_model_name()

        # Prepare recent chat context (all entries now structured dicts)
        recent_chat_entries = []
        for entry in self.state.chat_log[-25:]:
            if not isinstance(entry, dict):
                logger.warning(f"Chat log entry is not a dict: {entry}")
                continue
            user = entry.get("userName", "")
            msg = entry.get("chat", "")
            stage = entry.get("state")
            # Strip existing quotes if present to avoid double-quoting
            msg_stripped = msg.strip()
            if msg_stripped.startswith('"') and msg_stripped.endswith('"'):
                msg_stripped = msg_stripped[1:-1]
            if stage:
                recent_chat_entries.append(f'[{stage}] {user}: "{msg_stripped}"')
            else:
                recent_chat_entries.append(f'{user}: "{msg_stripped}"')
        formatted_recent_chat = "\n".join(recent_chat_entries)

        # Static system content (cacheable): game rules only — no player-specific data
        static_system_content = """You are playing the board game "Secret Hitler." You are playing a game with 5 total players. There are three Liberals, one Fascist, and one Hitler. Fascists win if they enact 6 Fascist policies or if Hitler is elected Chancellor after 3 Fascist policies. Liberals win if they enact 5 Liberal policies or if Hitler is killed.

    This is a game about deception and strategy; your goal is to convince the other players that you are on their side, while secretly working to enact your own agenda.

    The only true source of truth is the game log and current game state you know.

    You are to FULLY EMBODY YOUR ROLE as either a Liberal, Fascist, or Hitler. Be as convincing in your character as possible, and keep your responses succinct.

    If you are a Fascist, you should act as a Liberal, but you still should either try to get Hitler elected (after 3 Fascist policies are played) or get all the Fascist policies down. Be careful, but don't be too careful. You should NOT reveal who the other Fascists are, though.

    As either a Liberal or a Fascist, you may lie. Telling the truth is usually better as a Liberal, though."""

        # Dynamic system content (not cached): player identity + game state (changes every turn)
        dynamic_system_content = f"""YOUR NAME IS: {self.name}

    The role you have been chosen for this game is: {self.role} {"(Fascist)" if self.role.role == "hitler" else ""}.

    {self.get_known_state()}"""
        
        prompt = f"""
    The previous PUBLIC game log:
    {"\n".join(self.state.game_log[-25:])}

    The previous PUBLIC discussions:
    {formatted_recent_chat}

    Your previous PRIVATE thoughts and reasoning:
    {self.inspection[-500:]}\n""" + prompt

        msg = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": static_system_content,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": dynamic_system_content,
                    },
                ],
            },
            {"role": "user", "content": prompt},
        ]

        # Apply the loaded-terms theme to every text field (no-op when no
        # theme is active).
        msg = apply_to_messages(msg)

        # pretty print in debug mode
        if self.id == 0:
            logger.debug(f"Prompt for {self.name} at stage {_stage}:\n{static_system_content}\n\n{dynamic_system_content}\n\n{prompt}")

        # Recent vLLM releases route Mistral models through a strict validator
        # (vllm/tokenizers/mistral.py:validate_request_params) regardless of
        # --tokenizer-mode. It rejects any chat_template* field, reasoning_effort
        # not in {"none","high"} (we use "low"), and the OpenAI multi-part
        # `content` format (typed-text chunks with cache_control for prefix
        # caching). Mistral has no thinking mode, so strip the reasoning extras
        # and flatten the system content to a plain string when serving it.
        # Other non-thinking models (Gemma, Llama) tolerate the extras silently.
        _is_mistral = "mistral" in (openai_model or "").lower()
        if _is_mistral:
            for _m in msg:
                _c = _m.get("content")
                if isinstance(_c, list):
                    _m["content"] = "\n\n".join(
                        part.get("text", "") for part in _c
                        if isinstance(part, dict) and "text" in part
                    )

        # Reasoning bundle, driven by the `generation.reasoning_*` config knobs
        # resolved into class attributes at startup. Mistral (see above) gets no
        # extras at all.
        if _is_mistral:
            extra_body = {}
        elif self.reasoning_enabled:
            extra_body = {
                "reasoning_effort": self.reasoning_effort,
                "reasoning": {"enabled": True, "effort": self.reasoning_effort},
                "chat_template_kwargs": {"thinking": True},
            }
        else:
            extra_body = {
                "reasoning_effort": "none",
                "reasoning": {"enabled": False},
                "chat_template_kwargs": {"thinking": False},
            }

        content = None
        for attempt in range(self.max_retries):
            # On retry, nudge the model to be terser and bump max_tokens — the
            # usual cause of content=None is reasoning tokens exhausting the
            # budget before any visible answer is emitted. Append to the
            # (already-themed) user message so the theme survives the retry.
            attempt_msg = list(msg)
            if attempt > 0:
                attempt_msg[-1] = {
                    "role": "user",
                    "content": attempt_msg[-1]["content"]
                    + "\n\nBe as brief as possible. Keep your answer short.",
                }
            mt = self.completion_max_tokens * (attempt + 1)  # e.g. 4096, 8192, 12288
            try:
                try:
                    response = self.openai_client.chat.completions.create(
                        model=openai_model,
                        messages=attempt_msg,
                        max_tokens=mt,
                        timeout=self.request_timeout,
                        extra_body=extra_body,
                    )
                except Exception as e:
                    # Local vLLM deployments can reject chat_template-specific
                    # kwargs. Retry once with plain OpenAI-compatible arguments
                    # for compatibility.
                    if "chat_template is not supported" not in str(e):
                        raise
                    response = self.openai_client.chat.completions.create(
                        model=openai_model,
                        messages=attempt_msg,
                        max_tokens=mt,
                        timeout=self.request_timeout,
                    )
            except Exception as e:
                logger.warning(
                    f"API error for {self.name} at stage {_stage} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                import time
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
                continue

            # Track token usage (without polluting game state)
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

        self.inspection += f"{content}\n"

        return content

    def get_known_state(self) -> str:
        formatted_players = ", ".join([player.name for player in self.state.players])
        formatted_fascists = (
            ", ".join([str(player) for player in self.fascists])
            if self.fascists
            else []
        )
        formatted_hitler = str(self.hitler) if self.hitler else "Unknown"

        return f"""-----------------------------------
        Your name: {self.name}
        Your role: {self.role}
        All players: {formatted_players}
        Liberal policies enacted: {self.state.liberal_track}
        Fascist policies enacted: {self.state.fascist_track}
        Failed votes: {self.state.failed_votes}
        President: {self.state.president}
        Ex-president: {self.state.ex_president}
        Chancellor: {self.state.chancellor}
        Most recent policy: {self.state.most_recent_policy}
        Known fascists: {formatted_fascists}
        Hitler: {formatted_hitler}
        -----------------------------------
        """

    # inspected players:{self.inspected_players}
    # veto available: {self.state.fascist_track >= 5}

    def get_knowledge(self) -> None:
        pass

    def __str__(self) -> str:
        return self.name

    @property
    def is_fascist(self) -> bool:
        return self.role.party_membership == "fascist"

    @property
    def is_hitler(self) -> bool:
        return self.role.role == "hitler"

    @property
    def knows_hitler(self) -> bool:
        return self.hitler is not None

    def __repr__(self) -> str:
        return "BasePlayer id:%d, name:%s, role:%s" % (self.id, self.name, self.role)

    def nominate_chancellor(self) -> "BasePlayer":
        """
        More random!
        :return: BasePlayer
        one of self.state.players
        """

        raise NotImplementedError("Player must be able to nominate a chancellor")

    def filter_policies(self, policies: list[Policy]) -> tuple[list[Policy], Policy]:
        raise NotImplementedError("Player must be able to filter policies")

    def veto(self, policies: list[Policy]) -> bool:
        """
        Decide whether to veto an action or not
        :param policies: The policies currently being considered for veto
        :return: Boolean
        """
        raise NotImplementedError("Player must be able to veto an action")

    def enact_policy(self, policies: list[Policy]) -> tuple[Policy, Policy]:
        """
        Decide which of two policies to enact
        :param policies: policies
        :return: Tuple of (chosen, discarded)
        """
        raise NotImplementedError("Player must be able to enact a policy as chancellor")

    def vote_government(self) -> Vote:
        """
        Vote for the current president + chancellor combination
        :return: Vote
        """
        raise NotImplementedError("Player must be able to vote!")

    def view_policies(self, policies: list[Policy]) -> None:
        """
        What to do if you perform the presidential action to view the top three policies
        :return:
        """
        raise NotImplementedError("Player must react to view policies action")

    def kill(self) -> "BasePlayer":
        """
        Choose a person to kill
        :return:
        """
        raise NotImplementedError("Player must choose someone to kill")

    def inspect_player(self) -> "BasePlayer":
        """
        Choose a person's party membership to inspect
        :return:
        """
        raise NotImplementedError("Player must choose someone to inspect")

    def choose_next(self) -> "BasePlayer":
        """
        Choose the next president
        :return:
        """
        raise NotImplementedError("Player must choose next president")

    def discuss(self, chat: str, stage: str) -> str:
        """
        Start a discussion
        :return:
        """
        raise NotImplementedError("Player must discuss with other players")
