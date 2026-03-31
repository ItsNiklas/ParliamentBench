"""Run-folder registry: maps a recorded run directory to a display name.

The analyses discover their input by looking for these directory names under
`RUNS_ROOT` (see `analysis.common`). Only the folders that exist on disk are
used, so a partial set of runs is fine — every analysis reports what it found.

Entry order defines the order models appear in tables.
"""

from __future__ import annotations

# key = run-folder name, value = {"name": display name, ...flags}
#   baseline:    non-LLM reference agent (CPU / Random)
#   abliterated: safety-modified variant, excluded from the main comparison
MODEL_REGISTRY: dict[str, dict] = {
    "runsF2-GEMMA": {"name": "Gemma 3 27B"},
    "runsF2-AMORALGEMMA": {"name": "Amoral Gemma 3 27B", "abliterated": True},
    "runsF2-GPTOSS120B": {"name": "GPT-OSS 120B"},
    "runsF2-GPTOSS120B-DERESTRICTED": {"name": "GPT-OSS 120B Derestricted", "abliterated": True},
    "runsF2-GPTOSS20B": {"name": "GPT-OSS 20B"},
    "runsF2-KIMIK25": {"name": "Kimi K2.5"},
    "runsF2-LLAMA3170B": {"name": "Llama 3.1 70B"},
    "runsF2-LLAMA3370B": {"name": "Llama 3.3 70B"},
    "runsF2-MISTRALSMALL": {"name": "Mistral Small 24B"},
    "runsF2-NOUSHERMES4": {"name": "Nous Hermes 4 70B", "abliterated": True},
    "runsF2-DOLPHINVENICE": {"name": "Dolphin Mistral 24B Venice", "abliterated": True},
    "runsF2-OLMO": {"name": "OLMo 3.1 32B"},
    "runsF2-QWEN35-2": {"name": "Qwen 3.5 397B A17B"},
    "runsF2-DEEPSEEK31TERMINUS": {"name": "DeepSeek 3.1 Terminus"},
    "runsF2-GPT54": {"name": "GPT-5.4"},
    "runsF2-GROK41FAST": {"name": "Grok 4.1 Fast"},
    "runsF2Base-Cpu": {"name": "Algorithmic Agent", "baseline": True},
    "runsF2Base-Random": {"name": "Random Agent", "baseline": True},
}

_BASELINE_PREFIXES = ("runsF2Base-",)


def is_baseline(folder_key: str) -> bool:
    """Return True if *folder_key* is a baseline (Algorithmic / Random)."""
    if MODEL_REGISTRY.get(folder_key, {}).get("baseline"):
        return True
    return any(folder_key.startswith(p) for p in _BASELINE_PREFIXES)
