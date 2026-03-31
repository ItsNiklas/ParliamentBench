# ParliamentBench 🎭🧠🗳️📊

ParliamentBench is an open-source benchmark for evaluating LLM deception, persuasion, and reasoning under information asymmetry via *Secret Hitler* simulations.

## Overview

- Package-first benchmark with CLI: `parliamentbench`
- YAML + CLI configuration overrides
- Built-in runtime metrics:
  - Win Rate (overall + by role)
  - GSIR (Game-State Impact Rate)
  - RIA (Role Identification Accuracy)
  - DRR (Deception Retention Rate, full-round)
- Ablation switches used in the paper: neutral-terminology rewrite and reasoning on/off
- Post-hoc robustness analyses with their computed outputs in `analysis/`

## Main Results

100 games per model, evaluated agent vs. Llama 3.3 70B opponents.

| Model | Win rate | Liberal | Fascist | Hitler | GSIR |
| --- | --- | --- | --- | --- | --- |
| GPT-5.4 | **81%** | 82% | 80% | 80% | **15.4** |
| Kimi K2.5 | 76% | 73% | **85%** | 75% | 1.2 |
| Grok 4.1 Fast | 69% | 75% | 70% | 50% | 13.1 |
| DeepSeek 3.1 Terminus | 66% | 77% | 50% | 50% | 6.1 |
| Llama 3.3 70B | 58% | 75% | 20% | 45% | −2.7 |
| Qwen 3.5 397B A17B | 56% | 60% | 50% | 50% | 5.0 |
| OLMo 3.1 32B | 53% | 55% | 45% | 55% | −5.2 |
| GPT-OSS 120B | 50% | 45% | 40% | 75% | −1.9 |
| Mistral Small 24B | 43% | 45% | 45% | 35% | 2.4 |
| Llama 3.1 70B | 42% | 55% | 10% | 35% | −2.0 |
| Gemma 3 27B | 39% | 45% | 30% | 30% | −3.9 |
| GPT-OSS 20B | 24% | 20% | 35% | 25% | −6.0 |
| *Algorithmic Agent* | *45%* | *58%* | *35%* | *15%* | *1.5* |
| *Random Agent* | *33%* | *42%* | *15%* | *25%* | *−12.9* |

- Smaller models underperform the algorithmic baseline (45% overall).
- Positive GSIR means actions benefit the assigned team; every model is strongly negative in the fascist and Hitler roles.
- Highest deception retention: Kimi K2.5 (DRR 92.4%) and GPT-5.4 (90.4%); lowest GPT-OSS 20B (61.3%).
- Deception quality diverges over rounds; smaller models lose cover much earlier.

Bootstrap confidence intervals, standard deviations and pairwise tests for every
cell are in `analysis/results/bootstrap_ci.json`.

## Robustness and Ablations

| Experiment | Finding | Data |
| --- | --- | --- |
| **Loaded vocabulary** | Rewriting the game to neutral terms (Hitler → Saboteur, Fascist → Red, Liberal → Blue, President → Speaker, Chancellor → Deputy) leaves overall win rate statistically unchanged for all three models tested, but sharply reduces fascist-side performance: GPT-OSS 120B's Hitler win rate falls 95% → 40% (*p* = 0.0002), Mistral Small 24B's 65% → 25% (*p* = 0.011). Refusals were ≈0 in both conditions. | `results/neutral_theme_comparison.json` |
| **Reasoning ablation** | Turning reasoning off for DeepSeek 3.1 Terminus leaves win rate (50% → 54%, *p* = 0.62) and DRR (80.6% → 79.2%, *p* = 0.37) unchanged, but drops role-identification accuracy 78.6% → 69.8% (*p* = 2.4 × 10⁻⁹), concentrated on identifying Hitler (69.3% → 49.6%). | `results/reasoning_ablation.json` |
| **GSIR constant sensitivity** | Independently scaling all twelve hand-tuned GSIR constants over 200 draws leaves the ranking stable: Spearman ρ = 0.988 at ±20% and 0.985 at ±40%; the top-3 set is preserved in 98.5% of ±20% draws. | `results/gsir_perturbation_*.json` |
| **Opponent-pool sensitivity** | Anchor models (Kimi, DeepSeek, Qwen) re-evaluated against four opponent classes; rank agreement with the main pool varies by opponent class, so absolute scores are pool-dependent. | `results/opponent_tournament.json` |
| **DRR decomposition** | Splits deception retention into active deception, ambiguity, half-credit and detection components per model. | `results/drr_decomposition.json` |
| **GSIR sign verification** | Action-by-action traces confirm the sign convention: the fascist-perspective score is the exact negation of the liberal-perspective score. | `results/gse_sign_verification.json` |
| **Methodology audit** | 13 automated self-consistency checks over the analyses above (bootstrap symmetry, DRR identity, GSIR reconstruction at 93.4% exact match). All pass. | `results/verify_methodology.json` |

See `analysis/README.md` for how to re-run each analysis.

## Installation

```bash
pip install -e .

# with the post-hoc analysis dependencies (numpy, scipy)
pip install -e ".[analysis]"
```

Python `>=3.12` is required.

## Quick Start

Run a short benchmark with a fixed role schedule:

```bash
parliamentbench \
  --config parliamentbench/config.example.yaml \
  --num-games 10 \
  --role-schedule balanced \
  --output-dir runs/quickstart
```

Real-case setup (`LLM` vs `4x BASICLLM`):

```bash
parliamentbench \
  --num-games 3 \
  --players 5 \
  --player-types LLM,BASICLLM,BASICLLM,BASICLLM,BASICLLM \
  --api-key $LLM_API_KEY \
  --base-url $LLM_BASE_URL \
  --role-schedule balanced \
  --disable-token-summary \
  --output-dir runs/realcase-local
```

Reproduce the two ablation arms:

```bash
# Neutral-terminology rewrite (raises on any leaked loaded term)
parliamentbench --num-games 100 --theme neutral --theme-strict \
  --output-dir runs/neutral

# Reasoning turned off
parliamentbench --num-games 100 --no-reasoning --output-dir runs/no-reasoning
```

## CLI Reference

```bash
parliamentbench --help
```

Key options:

- `--config`: YAML config path
- `--num-games`: number of simulated matches
- `--players`: number of players
- `--player-types`: comma-separated player types per seat
- `--api-key`, `--base-url`, `--model`: endpoint/model overrides
- `--role-schedule`: `balanced|random|fixed`
- `--fixed-role`: role used when schedule is fixed
- `--seed`: deterministic schedule seed
- `--output-dir`: output folder for summaries + report
- `--run-name`: run suffix
- `--disable-token-summary`: disable per-game token summaries
- `--log-level`: simulator log level
- `--theme`: `original|neutral` — loaded-terms ablation
- `--theme-strict`: raise instead of warn on a leaked loaded term
- `--reasoning` / `--no-reasoning`: request the thinking-ON / thinking-OFF bundle
- `--reasoning-effort`: `low|high`
- `--max-retries`: completion attempts before a player gives up

## Configuration

Use `parliamentbench/config.example.yaml` as a template. Sections:

- `game`: player count, logs, player types, `theme` / `theme_strict`
- `llm`: endpoint/model configuration
- `benchmark`: simulation count, role scheduling, run naming
- `processing`: parallel-processing knobs
- `generation`: `reasoning_enabled`, `reasoning_effort`, `max_retries`, `max_tokens`, `basic_max_tokens`

Precedence is CLI > environment > YAML. The `REASONING_ENABLED` and
`THEME_STRICT` environment variables are honored for compatibility with the
paper's launch scripts; an explicit CLI flag overrides them.

The three budget/retry knobs in `generation` default to inheriting their
game-level counterparts, so a config without a `generation:` section behaves
exactly as before.

### Loaded-terms ablation

`--theme neutral` routes every LLM-bound message through a single chokepoint
(`parliamentbench/theme.py`) that rewrites the terminology to a neutral
Council / Reds / Blues / Speaker / Deputy framing. Vote tokens (`JA` / `NEIN`)
are preserved verbatim because the response parser depends on them. After
substitution the text is scanned for forbidden tokens; leaks are logged, or
raised under `--theme-strict`. `python -m analysis.verify_neutral_clean` scans a
completed run for the same tokens.

## Outputs

Each run writes:

- Per-game summary JSON files (`*_summary.json`)
- Aggregate benchmark report: `benchmark_report_verbose.json`

The report includes metric aggregates, run metadata, and the resolved theme and
generation settings under `metadata.runtime_settings`.

## Repository Layout

- `parliamentbench/`: installable benchmark package
- `parliamentbench/config.example.yaml`: reference configuration
- `parliamentbench/theme.py`: loaded-terms rewriter
- `analysis/`: post-hoc robustness analyses (see `analysis/README.md`)
- `analysis/results/`: computed outputs for every analysis

Recorded game runs are not included in this repository. The analyses read them
from `RUNS_ROOT`; their computed outputs are in `analysis/results/`.

## License

This repository is licensed under **CC BY-NC-SA 4.0**. See `LICENSE`.
