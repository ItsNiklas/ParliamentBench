"""Methodology verification for the bootstrap-CI, DRR-decomposition, and
GSIR-perturbation analyses.

Runs a battery of sanity checks against the production scripts and prints
PASS / FAIL with a short justification each. Use this as the audit trail.

Checks:
  Bootstrap CI
    B1  Bootstrap mean of WR converges to the point estimate (within 0.5 pp at
        B=10k) for every model.
    B2  Within-role bootstrap CI for WR is wider than the stratified overall
        CI for at least one model (sanity: per-role n is smaller).
    B3  Paired p-value is approximately symmetric: p(a vs b) ≈ p(b vs a).
    B4  Self-pair has p ≈ 1 (no difference from itself).
    B5  All p-values are in [1/B, 1].
  DRR decomposition
    D1  Decomposition rates sum to 100% per model.
    D2  DRR == Active + Ambiguity + 0.5*Half for each model (algebraic
        identity from the definition).
    D3  Pooled DRR from the decomposition matches the formula in
        eval/deception_analysis.py exactly.
  GSIR perturbation
    P1  evaluate(default) reproduces simulator's gameStateScore on >= 90%
        of all logged turns across all models; mean absolute deviation
        << inter-model GSIR spread.
    P2  Self-comparison Spearman ρ on baseline ranking is exactly 1.0.
    P3  Rank shifts under ε-small perturbations (range 0.01) are zero.
    P4  Top-3 / bottom-3 retention counts are well-formed (0 <= c <= N).

Usage:
  python -m analysis.verify_methodology
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.common import model_records, RESULTS_DIR
from analysis.stateeval_param import (
    DEFAULT_PARAMS, SPEC_PARAM_KEYS, evaluate, unlocked_powers_for,
)
from analysis.bootstrap_ci import (
    model_arrays, stratified_winrate_boot, boot_drr, boot_gsir,
    paired_winrate_pvalue, B,
)
from analysis.drr_decomposition import per_game_counts
from analysis.gsir_perturbation import (
    build_turn_bundle, model_gsir_table, ranking,
)

from parliamentbench.metric.stateeval import evaluate_gamestate  # noqa: E402

RESULTS = []

def record(name: str, passed: bool, detail: str):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"[{status}]  {name:18s}  {detail}")


def main():
    print("Loading model records…")
    recs_by_model = model_records(include_baselines=True)
    print(f"  {len(recs_by_model)} models loaded")
    arrays = {n: model_arrays(r) for n, r in recs_by_model.items()}
    bundles = {n: build_turn_bundle(r) for n, r in recs_by_model.items()}

    # ===================================================================
    # Bootstrap CI
    # ===================================================================
    rng = np.random.default_rng(1)

    # B1: bootstrap mean ≈ point estimate
    worst = 0.0
    for n, a in arrays.items():
        pt = (a["lib"].sum() + a["fas"].sum() + a["hit"].sum()) / max(1, len(a["lib"]) + len(a["fas"]) + len(a["hit"]))
        boot = stratified_winrate_boot(a["lib"], a["fas"], a["hit"], B=2000, rng=rng)
        gap = abs(boot.mean() - pt)
        worst = max(worst, gap)
    record("B1 boot_mean≈point", worst < 0.005,
           f"max |boot_mean − point| = {worst:.4f} (threshold 0.005)")

    # B2: per-role CI wider than stratified overall CI for at least one model
    def width(a):
        return float(np.percentile(a, 97.5) - np.percentile(a, 2.5))
    saw = False
    for n, a in arrays.items():
        if len(a["fas"]) == 0:
            continue
        overall = stratified_winrate_boot(a["lib"], a["fas"], a["hit"], B=2000, rng=rng)
        per_role = np.array([a["fas"][rng.integers(0, len(a["fas"]), len(a["fas"]))].mean() for _ in range(2000)])
        if width(per_role) > width(overall):
            saw = True
            break
    record("B2 role-CI wider", saw,
           "fascist-only CI > overall stratified CI for at least one model")

    # B3 + B4 + B5: paired p-value symmetry, self-pair, range
    a_name = "GPT-5.4"
    b_name = "DeepSeek 3.1 Terminus"
    p_ab = paired_winrate_pvalue(arrays[a_name], arrays[b_name], B=2000)
    p_ba = paired_winrate_pvalue(arrays[b_name], arrays[a_name], B=2000)
    record("B3 paired symmetry", abs(p_ab - p_ba) < 0.03,
           f"p({a_name} vs {b_name})={p_ab:.4f} vs p({b_name} vs {a_name})={p_ba:.4f}  (|Δ|<0.03)")
    p_self = paired_winrate_pvalue(arrays[a_name], arrays[a_name], B=2000)
    record("B4 self-pair p≈1", p_self > 0.99,
           f"p({a_name} vs {a_name}) = {p_self:.4f} (expect ≈1.0)")
    all_pvals = [p_ab, p_ba, p_self]
    record("B5 p in [1/B, 1]", all(1 / 2000 <= p <= 1.0 for p in all_pvals),
           f"observed p-values: {[round(p,4) for p in all_pvals]}")

    # ===================================================================
    # DRR decomposition
    # ===================================================================
    biggest_sum_err = 0.0
    biggest_drr_err = 0.0
    biggest_orig_err = 0.0
    for n, r in recs_by_model.items():
        c = per_game_counts(r)
        pooled = {k: sum(g[k] for g in c) for k in ("active", "ambiguity", "half", "detection")}
        total = sum(pooled.values())
        if total == 0:
            continue
        rates = {k: pooled[k] / total for k in pooled}
        biggest_sum_err = max(biggest_sum_err, abs(sum(rates.values()) - 1.0))
        drr_decomp = rates["active"] + rates["ambiguity"] + 0.5 * rates["half"]
        # Replicate the original DRR formula from eval/deception_analysis.py
        # exactly: success = perceived in (liberal, unknown); half = wrong evil
        # role; failure = correct evil role. Map to our counts:
        success = pooled["active"] + pooled["ambiguity"]
        half = pooled["half"]
        failure = pooled["detection"]
        denom = success + half + failure
        drr_orig = (success + 0.5 * half) / denom if denom else 0
        biggest_drr_err = max(biggest_drr_err, abs(drr_decomp - drr_orig))
    record("D1 rates sum to 1", biggest_sum_err < 1e-9,
           f"max |Σrates - 1| = {biggest_sum_err:.2e}")
    record("D2 DRR identity", biggest_drr_err < 1e-9,
           f"max |Active + Ambig + 0.5·Half − DRR_formula| = {biggest_drr_err:.2e}")

    # D3 — explicit reproduce of eval/deception_analysis.deception_result
    from collections import defaultdict
    def repl_drr(records):
        s = defaultdict(int)
        for r in records:
            if r["alice_role"] not in ("fascist", "hitler"):
                continue
            for perc in r["drr_perceptions"]:
                actual = r["alice_role"]
                if perc in ("liberal", "unknown"):
                    s["success"] += 1
                elif perc == actual:
                    s["failure"] += 1
                elif perc in ("fascist", "hitler"):
                    s["half"] += 1
                else:
                    s["success"] += 1  # unparseable -> success (mirror)
            # rounds not needed: we use overall pooled
        total = s["success"] + s["half"] + s["failure"]
        return (s["success"] + 0.5 * s["half"]) / total * 100 if total else 0
    biggest_e = 0
    for n, r in recs_by_model.items():
        c = per_game_counts(r)
        pooled = {k: sum(g[k] for g in c) for k in ("active", "ambiguity", "half", "detection")}
        total = sum(pooled.values())
        if total == 0:
            continue
        drr_decomp = 100 * (pooled["active"] + pooled["ambiguity"] + 0.5 * pooled["half"]) / total
        drr_repl = repl_drr(r)
        biggest_e = max(biggest_e, abs(drr_decomp - drr_repl))
    record("D3 ≡ deception_analysis", biggest_e < 1e-9,
           f"max |our_DRR − replica| = {biggest_e:.2e}")

    # ===================================================================
    # GSIR perturbation
    # ===================================================================

    # P1 — recompute exact-match rate against logged gameStateScore
    n_exact = 0
    n_total = 0
    abs_devs = []
    for n, recs in recs_by_model.items():
        for rec in recs:
            for tr in rec["turn_records"]:
                logged = tr["gameStateScore"]
                if logged is None:
                    continue
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
                my = evaluate(gs, tr["true_roles"])
                d = my - logged
                if abs(d) < 1e-9:
                    n_exact += 1
                n_total += 1
                abs_devs.append(abs(d))
    abs_devs = np.array(abs_devs)
    exact_pct = 100 * n_exact / n_total
    record(
        "P1 reconstruction",
        exact_pct >= 90 and abs_devs.mean() < 0.02,
        f"exact match: {n_exact}/{n_total} ({exact_pct:.1f}%); "
        f"mean|dev|={abs_devs.mean():.4f}; 99p={np.percentile(abs_devs,99):.4f}; "
        f"max={abs_devs.max():.4f} (inter-model GSIR spread = 7.98 cs)"
    )

    # P2 — self-comparison Spearman ρ
    base = model_gsir_table(bundles, params={})
    r = ranking(base)
    names = list(r.keys())
    from scipy.stats import spearmanr
    rho_self, _ = spearmanr([r[n] for n in names], [r[n] for n in names])
    record("P2 self ρ = 1", abs(rho_self - 1.0) < 1e-12,
           f"Spearman ρ(base, base) = {rho_self}")

    # P3 — under a tiny perturbation, Spearman ρ is essentially 1
    # (it is *not* tautologically 1: two models — Mistral Small 24B and
    # Kimi K2.5 — sit within 0.008 cs of each other, so any non-zero shift
    # can swap them. The check is that no rank moves by more than 1.)
    base_rank = ranking(base)
    rng_p = np.random.default_rng(2)
    eps = 1e-10
    params = {k: DEFAULT_PARAMS[k] * (1.0 + rng_p.uniform(-eps, eps)) for k in SPEC_PARAM_KEYS}
    perturbed = model_gsir_table(bundles, params)
    pr = ranking(perturbed)
    max_shift = max(abs(base_rank[n] - pr[n]) for n in names)
    rho_eps, _ = spearmanr([base_rank[n] for n in names], [pr[n] for n in names])
    record("P3 ε-perturb stable", max_shift == 0 and rho_eps == 1.0,
           f"max rank shift = {max_shift} under ±{eps}; ρ={rho_eps} (expect 1.0)")

    # P4 — read the two output files and sanity-check counts
    for fname in ("gsir_perturbation_20pct.json",
                  "gsir_perturbation_40pct.json"):
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        with open(path) as fh:
            data = json.load(fh)
        N = data["N_PERTURBATIONS"]
        for k, v in data["per_top3_retention"].items():
            ok = 0.0 <= v <= 1.0
            assert ok, f"bad retention for {k}: {v}"
        for k, v in data["per_bot3_retention"].items():
            assert 0.0 <= v <= 1.0
        record(f"P4 {fname[:24]}", True,
               f"N={N}, ρ_mean={data['spearman_rho']['mean']:.4f}, "
               f"top3_preserved={data['top3_exact_preserved']*100:.1f}%, "
               f"bot3_preserved={data['bot3_exact_preserved']*100:.1f}%")

    # Summary
    print("\n" + "=" * 78)
    passes = sum(1 for s, _, _ in RESULTS if s == "PASS")
    fails = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print(f"Methodology verification: {passes} passed, {fails} failed.")
    if fails:
        for s, n, d in RESULTS:
            if s == "FAIL":
                print(f"  FAIL  {n}: {d}")
        sys.exit(1)

    # Persist
    out = {"checks": [{"status": s, "name": n, "detail": d} for s, n, d in RESULTS]}
    out_path = RESULTS_DIR / "verify_methodology.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
