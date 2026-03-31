"""Resolution of config settings into global / class-level runtime state.

Both entry points (`parliamentbench.benchmark` and the single-game
`parliamentbench.game_engine` `__main__`) call `apply_runtime_settings` so the
theme and generation knobs behave identically either way.
"""

from __future__ import annotations

import os
from typing import Any

from . import theme as theme_mod
from .config_loader import Config
from .players import BasePlayer


def apply_runtime_settings(config: Config) -> dict[str, Any]:
    """Resolve the theme and generation knobs into global/class-level state.

    Precedence is CLI > environment > YAML. Any CLI overrides have already been
    written onto *config* by the caller, so only the environment overrides are
    applied here.

    Returns the resolved settings so callers can log them or record them in a
    run report.
    """
    # Loaded-terms theme (no-op when config.theme is None).
    theme_mod.set_theme(config.theme)
    theme_mod.THEME_STRICT = bool(config.theme_strict) or theme_mod.THEME_STRICT

    gen = config.effective_generation()

    # REASONING_ENABLED still wins over YAML for parity with the ablation
    # launchers, but an explicit --reasoning/--no-reasoning flag wins over both
    # (the CLI layer clears the variable when it is passed).
    env_reasoning = os.environ.get("REASONING_ENABLED")
    if env_reasoning is not None:
        gen["reasoning_enabled"] = env_reasoning != "0"

    BasePlayer.enable_parallel_processing = config.enable_parallel
    BasePlayer.reasoning_enabled = gen["reasoning_enabled"]
    BasePlayer.reasoning_effort = gen["reasoning_effort"]
    BasePlayer.max_retries = gen["max_retries"]
    BasePlayer.completion_max_tokens = gen["completion_max_tokens"]
    BasePlayer.basic_max_tokens = gen["basic_max_tokens"]

    return {
        "theme": theme_mod.ACTIVE_THEME,
        "theme_strict": theme_mod.THEME_STRICT,
        **gen,
    }
