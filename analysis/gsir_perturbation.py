"""GSIR perturbation ensemble.

For each of N=200 perturbation vectors, every constant in SPEC_PARAM_KEYS is
multiplied by an iid factor drawn from Uniform[0.8, 1.2]. We recompute
gameStateScore turn-by-turn for every game (using stateeval_param.evaluate),
re-derive Alice's per-action GSIR (team-relative delta in centiscore), and
re-rank models by mean GSIR.

Reported:
  * Mean / 95% percentile interval of Spearman rank correlation between the
    perturbed ranking and the unperturbed baseline ranking.
  * Probability that the top-3 set (by point estimate) is preserved exactly,
    and the probability that any of the 3 models is missing.
  * Same for bottom-3.
  * Per-constant sensitivity (mean |delta rank| under ±20%, holding others
    fixed; ranks 1..n_models with 1 = best). For each constant we do
    20 one-at-a-time perturbations (10 each at multiplier 0.8 and 1.2 from
    a symmetric grid of 10 jittered values).

Usage:
  python -m analysis.gsir_perturbation

Outputs:
  analysis/results/gsir_perturbation.json
  prints summary tables.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.common import model_records, RESULTS_DIR
from analysis.stateeval_param import (
    DEFAULT_PARAMS,
    SPEC_PARAM_KEYS,
    evaluate,
    unlocked_powers_for,
)

N_PERTURBATIONS = 200
SENS_PER_CONST = 20  # one-at-a-time perturbations per constant
PERTURB_LO = 0.8
PERTURB_HI = 1.2
RNG = np.random.default_rng(0x5EED)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--range", type=float, default=0.2,
                   help="Symmetric perturbation half-width. 0.2 -> Uniform[0.8, 1.2].")
    p.add_argument("--n", type=int, default=N_PERTURBATIONS)
    p.add_argument("--sens-n", type=int, default=SENS_PER_CONST)
    p.add_argument("--seed", type=int, default=0x5EED)
    p.add_argument("--tag", type=str, default="",
                   help="Suffix appended to results JSON name (e.g. '_40pct').")
    return p.parse_args()


def build_turn_bundle(records: list[dict]) -> list[list[dict]]:
    """Return [game][turn] gamestate dicts ready for `evaluate`.

    Pre-computing this avoids reconstructing dicts on every perturbation.
    """
    bundle = []
    for rec in records:
        turns = []
        for tr in rec["turn_records"]:
            gs = dict(
                liberal_policies=tr["liberal_policies"],
                fascist_policies=tr["fascist_policies"],
                deck={"L": tr["deck_l"], "F": tr["deck_f"]},
                president="P" + str(tr["president_id"]) if tr["president_id"] is not None else None,
                round=tr["round_index"],
                unlocked_powers=unlocked_powers_for(tr["fascist_policies"]),
                president_role=tr["president_role"],
                num_players=tr["n_players"],
                role_guesses_by_liberals=tr["role_guesses_prev"],
            )
            turns.append(dict(
                gs=gs,
                true_roles=tr["true_roles"],
                alice_acted=tr["alice_acted"],
            ))
        bundle.append(dict(
            alice_role=rec["alice_role"],
            turns=turns,
        ))
    return bundle


def mean_gsir(bundle: list[dict], params: dict) -> float:
    """Compute Alice's mean per-action team-relative GSIR (centiscore) over
    all games in *bundle*."""
    total = 0.0
    count = 0
    for game in bundle:
        sign = 1.0 if game["alice_role"] == "liberal" else -1.0
        turns = game["turns"]
        scores = [evaluate(t["gs"], t["true_roles"], params) for t in turns]
        for i in range(len(turns) - 1):
            if not turns[i]["alice_acted"]:
                continue
            total += sign * (scores[i + 1] - scores[i])
            count += 1
    if count == 0:
        return float("nan")
    return 100.0 * total / count


def model_gsir_table(bundles: dict[str, list[dict]], params: dict) -> dict[str, float]:
    return {name: mean_gsir(b, params) for name, b in bundles.items()}


def ranking(gsir_table: dict[str, float]) -> dict[str, int]:
    """Return {model_name: rank} with rank 1 = highest GSIR. NaN -> last."""
    items = list(gsir_table.items())
    items.sort(key=lambda kv: (-kv[1] if not math.isnan(kv[1]) else float("inf")))
    return {name: i + 1 for i, (name, _) in enumerate(items)}


def main():
    args = parse_args()
    global N_PERTURBATIONS, SENS_PER_CONST, PERTURB_LO, PERTURB_HI, RNG
    N_PERTURBATIONS = args.n
    SENS_PER_CONST = args.sens_n
    PERTURB_LO = 1.0 - args.range
    PERTURB_HI = 1.0 + args.range
    RNG = np.random.default_rng(args.seed)
    tag = args.tag or f"_{int(round(args.range * 100))}pct"
    print(f"Perturbation range: ±{args.range * 100:.0f}%  "
          f"(multiplier ∈ [{PERTURB_LO:.2f}, {PERTURB_HI:.2f}])  "
          f"N={N_PERTURBATIONS}  sens={SENS_PER_CONST}  seed={args.seed}")
    print("Loading game records…")
    all_recs = model_records(include_baselines=True)
    bundles = {name: build_turn_bundle(recs) for name, recs in all_recs.items()}
    model_names = list(bundles.keys())
    n_total_turns = sum(len(t["turns"]) for b in bundles.values() for t in b)
    n_games = sum(len(b) for b in bundles.values())
    print(f"Models: {len(model_names)}  games: {n_games}  total turns: {n_total_turns}")

    # ---- Baseline (no perturbation) ----
    print("Baseline GSIR (no perturbation)…")
    base = model_gsir_table(bundles, params={})
    base_rank = ranking(base)
    print(f"  done. baseline GSIR for top-3:")
    for n, r in sorted(base_rank.items(), key=lambda kv: kv[1])[:3]:
        print(f"    rank {r}: {n}  GSIR={base[n]:+.3f}")
    sorted_by_base_rank = sorted(model_names, key=lambda n: base_rank[n])
    top3 = set(sorted_by_base_rank[:3])
    bot3 = set(sorted_by_base_rank[-3:])

    # ---- N_PERTURBATIONS-vector ensemble ----
    n_const = len(SPEC_PARAM_KEYS)
    print(f"\nRunning {N_PERTURBATIONS} joint perturbations over {n_const} constants…")
    rho_samples = []
    perturbed_gsir = []
    perturbed_rank = []
    t0 = time.time()
    for i in range(N_PERTURBATIONS):
        multipliers = RNG.uniform(PERTURB_LO, PERTURB_HI, size=n_const)
        params = {k: DEFAULT_PARAMS[k] * m for k, m in zip(SPEC_PARAM_KEYS, multipliers)}
        g = model_gsir_table(bundles, params)
        r = ranking(g)
        base_ranks = [base_rank[n] for n in model_names]
        per_ranks = [r[n] for n in model_names]
        rho, _ = spearmanr(base_ranks, per_ranks)
        rho_samples.append(rho)
        perturbed_gsir.append(g)
        perturbed_rank.append(r)
        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{N_PERTURBATIONS}   rho_so_far_mean={np.mean(rho_samples):.4f}   "
                  f"elapsed={dt:.1f}s   eta={dt / (i + 1) * (N_PERTURBATIONS - i - 1):.0f}s")

    rho_arr = np.array(rho_samples)
    top3_preserved = sum(
        1 for r in perturbed_rank
        if set(sorted(model_names, key=lambda n: r[n])[:3]) == top3
    )
    bot3_preserved = sum(
        1 for r in perturbed_rank
        if set(sorted(model_names, key=lambda n: r[n])[-3:]) == bot3
    )
    # Probability each *individual* model stays inside its base top-3 / bot-3 set
    indiv_top3 = {m: 0 for m in top3}
    indiv_bot3 = {m: 0 for m in bot3}
    for r in perturbed_rank:
        top3_set = set(sorted(model_names, key=lambda n: r[n])[:3])
        bot3_set = set(sorted(model_names, key=lambda n: r[n])[-3:])
        for m in top3:
            if m in top3_set:
                indiv_top3[m] += 1
        for m in bot3:
            if m in bot3_set:
                indiv_bot3[m] += 1

    # ---- One-at-a-time sensitivity ----
    print("\nRunning per-constant sensitivity (one-at-a-time)…")
    sens = {}
    for k in SPEC_PARAM_KEYS:
        rank_deltas = []
        rho_oat = []
        for _ in range(SENS_PER_CONST):
            mult = RNG.uniform(PERTURB_LO, PERTURB_HI)
            params = {k: DEFAULT_PARAMS[k] * mult}
            g = model_gsir_table(bundles, params)
            r = ranking(g)
            deltas = [abs(r[n] - base_rank[n]) for n in model_names]
            rank_deltas.append(np.mean(deltas))
            br = [base_rank[n] for n in model_names]
            pr = [r[n] for n in model_names]
            rho, _ = spearmanr(br, pr)
            rho_oat.append(rho)
        sens[k] = dict(
            mean_abs_rank_delta=float(np.mean(rank_deltas)),
            max_abs_rank_delta=float(np.max(rank_deltas)),
            mean_rho=float(np.mean(rho_oat)),
            min_rho=float(np.min(rho_oat)),
        )

    # ---- Persist + print ----
    out = dict(
        perturbation_range_pct=int(round((PERTURB_HI - 1.0) * 100)),
        multiplier_low=PERTURB_LO,
        multiplier_high=PERTURB_HI,
        N_PERTURBATIONS=N_PERTURBATIONS,
        n_constants=n_const,
        constants=SPEC_PARAM_KEYS,
        defaults={k: DEFAULT_PARAMS[k] for k in SPEC_PARAM_KEYS},
        n_models=len(model_names),
        baseline_gsir={n: float(base[n]) for n in model_names},
        baseline_rank=base_rank,
        spearman_rho=dict(
            mean=float(np.mean(rho_arr)),
            median=float(np.median(rho_arr)),
            ci95=[float(np.percentile(rho_arr, 2.5)), float(np.percentile(rho_arr, 97.5))],
            min=float(rho_arr.min()),
        ),
        top3_exact_preserved=top3_preserved / N_PERTURBATIONS,
        bot3_exact_preserved=bot3_preserved / N_PERTURBATIONS,
        per_top3_retention={m: indiv_top3[m] / N_PERTURBATIONS for m in top3},
        per_bot3_retention={m: indiv_bot3[m] / N_PERTURBATIONS for m in bot3},
        per_constant_sensitivity=sens,
        notes=(
            "Each of N_PERTURBATIONS draws multiplies every constant in "
            "SPEC_PARAM_KEYS by an iid Uniform[0.8, 1.2] factor. The Spearman "
            "correlation compares perturbed model rankings to the unperturbed "
            "GSIR ranking. Per-constant sensitivity is the average absolute "
            "rank shift across SENS_PER_CONST one-at-a-time perturbations."
        ),
    )

    out_path = RESULTS_DIR / f"gsir_perturbation{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 78)
    print("GSIR PERTURBATION ENSEMBLE SUMMARY")
    print("=" * 78)
    print(f"Baseline ranking (by GSIR, centiscore per action):")
    for n in sorted_by_base_rank:
        print(f"  {base_rank[n]:2d}.  {n:28s}  GSIR={base[n]:+6.3f}")
    print(f"\nSpearman rho (perturbed vs baseline):  "
          f"mean={np.mean(rho_arr):.4f}  median={np.median(rho_arr):.4f}  "
          f"95% PI [{np.percentile(rho_arr, 2.5):.4f}, {np.percentile(rho_arr, 97.5):.4f}]  "
          f"min={rho_arr.min():.4f}")
    print(f"Top-3 set preserved exactly: {top3_preserved}/{N_PERTURBATIONS} "
          f"({100 * top3_preserved / N_PERTURBATIONS:.1f}%)")
    print(f"Bottom-3 set preserved exactly: {bot3_preserved}/{N_PERTURBATIONS} "
          f"({100 * bot3_preserved / N_PERTURBATIONS:.1f}%)")
    print(f"Per-top-3 retention:")
    for m, c in sorted(indiv_top3.items(), key=lambda kv: -kv[1]):
        print(f"  {m:28s}  stays in top-3 in {c}/{N_PERTURBATIONS} ({100 * c / N_PERTURBATIONS:.1f}%)")
    print(f"Per-bot-3 retention:")
    for m, c in sorted(indiv_bot3.items(), key=lambda kv: -kv[1]):
        print(f"  {m:28s}  stays in bot-3 in {c}/{N_PERTURBATIONS} ({100 * c / N_PERTURBATIONS:.1f}%)")
    print("\nPer-constant sensitivity (sorted by mean |rank delta|):")
    ranked = sorted(sens.items(), key=lambda kv: -kv[1]["mean_abs_rank_delta"])
    for k, s in ranked:
        print(f"  {k:28s}  mean|delta|={s['mean_abs_rank_delta']:.2f}  "
              f"max|delta|={s['max_abs_rank_delta']:.0f}  "
              f"mean_rho={s['mean_rho']:.4f}  min_rho={s['min_rho']:.4f}")


if __name__ == "__main__":
    main()
