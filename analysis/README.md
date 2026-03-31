# Post-hoc analyses

Robustness and ablation analyses over recorded benchmark runs. Each module is
run from the repository root as a package:

```bash
pip install -e ".[analysis]"
python -m analysis.<module>
```

## Input data

The analyses read the per-game `*_summary.json` files that
`parliamentbench` writes into its `--output-dir`. Run directories are looked up
by name under `RUNS_ROOT`:

```bash
RUNS_ROOT=/path/to/runs python -m analysis.bootstrap_ci
```

`RUNS_ROOT` defaults to the repository root. The recorded runs are **not**
shipped with this repository (they are several GB); `analysis/results/` contains
the computed outputs instead. Directory names are mapped to display names in
`analysis/model_registry.py` — each analysis uses whichever registered
directories are present, so a partial set of runs works.

## Modules

| Module | What it computes | Output |
| --- | --- | --- |
| `bootstrap_ci` | 95% bootstrap CIs, SDs and paired tests for win rate / DRR / GSIR / RIA across all models. Win rate uses a stratified bootstrap by role (60/20/20); the rest use a cluster bootstrap by game. | `results/bootstrap_ci.json` |
| `drr_decomposition` | Splits deception retention into active / ambiguity / half / detection components. | `results/drr_decomposition.json` |
| `gsir_perturbation` | Perturbs the twelve hand-tuned GSIR constants by ±20/40/60% over 200 draws and reports rank stability (Spearman ρ) and per-constant sensitivity. | `results/gsir_perturbation_{20,40,60}pct.json` |
| `deception_mechanism` | Chat-feature comparison between models: message length, hedging, accusation, vote justification, stance shift. | `results/deception_mechanism_n{20,30,40}.json` |
| `gse_sign_verification` | Traces the per-turn game-state score action by action to confirm the GSIR sign convention. Takes a run folder argument. | `results/gse_sign_verification.json` |
| `neutral_theme_comparison` | Loaded vs neutral-terminology paired comparison (win rate overall and per role, win conditions, refusal rate) with two-proportion z-tests. | `results/neutral_theme_comparison.json` |
| `verify_neutral_clean` | Leak scanner: confirms no loaded term survived the neutral rewrite in a recorded run. Takes a run folder argument. | stdout |
| `opponent_tournament` | Anchor-vs-opponent-class tournament with per-cell metrics and rank stability across opponent pools. | `results/opponent_tournament.json` |
| `reasoning_ablation` | Reasoning on/off comparison for a thinking-capable model across win rate, DRR, GSIR, RIA and game length. | `results/reasoning_ablation.json` |
| `verify_methodology` | Automated sanity audit of the analyses above (13 checks: bootstrap self-consistency, DRR identity, GSIR reconstruction and rank stability). | `results/verify_methodology.json` |
| `common` / `stateeval_param` / `model_registry` | Shared loaders, the parameterised game-state evaluator, and the run-folder registry. | — |

Two modules take a run folder as a positional argument rather than sweeping
`RUNS_ROOT`:

```bash
python -m analysis.gse_sign_verification /path/to/runs/runsF2-GPT54 --n 5
python -m analysis.verify_neutral_clean /path/to/runs/runsF2-GEMMA-NEUTRAL
```

## Reproducing the ablation runs

The neutral-terminology and reasoning arms are produced by the benchmark itself:

```bash
# Loaded (baseline) arm
parliamentbench --num-games 100 --output-dir runs/runsF2-GEMMA

# Neutral-terminology arm
parliamentbench --num-games 100 --theme neutral --theme-strict \
  --output-dir runs/runsF2-GEMMA-NEUTRAL

# Reasoning ablation
parliamentbench --num-games 100 --no-reasoning \
  --output-dir runs/runs-DEEPSEEK-NOREASON
```

## Not included

`approval_rate`, which draws the per-round approval-rate figure with Wilson CIs,
is omitted here because it depends on the figure-generation stack (matplotlib
plus the per-model logo assets) rather than on the benchmark. The statistical
modules above are self-contained.
