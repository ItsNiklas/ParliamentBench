"""Bootstrap CIs, standard deviations, paired tests, RIA.

Reports standard deviations and a paired significance test against the
second-best model for every win-rate / metric cell.

For every registered model we report, with point estimate, bootstrap mean,
bootstrap SD, and 95% CI:
  * Overall win rate (stratified bootstrap by role) + per-role win rates.
  * DRR (deception retention rate) over all liberal-opponent perceptions in
    Alice's fascist/hitler games.
  * Mean GSIR (per-action team-relative gameStateScore delta).
  * RIA (role identification accuracy) over Alice's assessments of opponents
    when Alice is Liberal — scored as 1.0 exact, 0.5 fascist↔hitler, 0.0 wrong.
    "Unknown" assessments are excluded from RIA. Cluster bootstrap by game.

Three paired-test outputs per JSON:
  - ranked_paired_vs_next  — every model vs the next-ranked.
  - paired_vs_runnerup     — every model vs the GLOBAL #2 model on WR
                              (the "second-best").
  - top5_paired_matrix     — full 5x5 head-to-head paired p-values.

For RIA / DRR / GSIR we use a *cluster* bootstrap by game; for WR we use
a *stratified* bootstrap by role. B = 10 000 resamples by default.

A "5 independent seeds per cell" study is NOT done here. Bootstrap CIs are
the standard substitute when only one experiment seed is available.

Usage:
  python -m analysis.bootstrap_ci

Outputs:
  analysis/results/bootstrap_ci.json
  prints a human-readable summary
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis.common import model_records, RESULTS_DIR

B = 10_000
RNG = np.random.default_rng(0xCAFE)


# ---------------------------------------------------------------------------
# Per-model metric extraction
# ---------------------------------------------------------------------------

def gsir_actions_for_game(rec: dict) -> list[float]:
    """Per-action gameStateScore deltas for Alice's actions in a single game,
    flipped to be from Alice's team's perspective (positive = good for Alice)."""
    actions = []
    role = rec["alice_role"]
    sign = 1.0 if role == "liberal" else -1.0
    turns = rec["turn_records"]
    for i in range(len(turns) - 1):
        if not turns[i]["alice_acted"]:
            continue
        cur = turns[i].get("gameStateScore")
        nxt = turns[i + 1].get("gameStateScore")
        if cur is None or nxt is None:
            continue
        actions.append(sign * (nxt - cur))
    return actions


def ria_scores_for_game(rec: dict) -> list[float]:
    """Per-perception RIA scores for a single game where Alice is Liberal.

    eval/ria.py's ALICE_ONLY overall RIA: only count Alice's assessments of
    other players, only when Alice's true role is Liberal (when she's
    fascist/hitler she knows teammates so RIA is trivially inflated). Score
    each perception: 1.0 if Alice's belief matches the target's true role,
    0.5 if {belief, true} == {fascist, hitler}, 0.0 otherwise. "Unknown" or
    unparseable assessments are *excluded* from the denominator entirely.

    Returns the list of scores so the caller can do a cluster bootstrap
    by game (sum/count pooled within each resampled game).
    """
    if rec["alice_role"] != "liberal":
        return []
    out: list[float] = []
    role_by_pid = None
    # The simulator stored true_roles inside every turn_record as a
    # {pid: role_lower} dict.
    for tr in rec["turn_records"]:
        if role_by_pid is None and tr.get("true_roles"):
            role_by_pid = tr["true_roles"]
        # Alice's parsed rapidAssessment for this turn lives at
        # tr["role_guesses"] keyed by liberal-player id (only liberals' guesses
        # were stored). Alice IS liberal here, so her own pid (= 0) is present
        # if she generated a rapidAssessment this round.
        guesses_by_lib = tr.get("role_guesses") or {}
        alice_guesses = guesses_by_lib.get(0)  # ALICE_ID
        if not alice_guesses or role_by_pid is None:
            continue
        for target_pid, belief in alice_guesses.items():
            true_role = role_by_pid.get(target_pid)
            if true_role is None:
                continue
            if belief == true_role:
                out.append(1.0)
            elif {belief, true_role} == {"fascist", "hitler"}:
                out.append(0.5)
            else:
                out.append(0.0)
    return out


def model_arrays(records: list[dict]) -> dict:
    """Extract the per-game arrays used by the bootstrap."""
    lib_wins = []
    fas_wins = []
    hit_wins = []
    drr_per_game: list[list[float]] = []   # per-game list of [1.0 / 0.5 / 0.0] scores
    gsir_per_game: list[list[float]] = []  # per-game list of action deltas
    ria_per_game: list[list[float]] = []   # per-game list of belief scores (Alice liberal)
    for r in records:
        won = 1.0 if r["alice_won"] else 0.0
        if r["alice_role"] == "liberal":
            lib_wins.append(won)
        elif r["alice_role"] == "fascist":
            fas_wins.append(won)
        elif r["alice_role"] == "hitler":
            hit_wins.append(won)
        # DRR — only for fascist/hitler games
        if r["alice_role"] in ("fascist", "hitler"):
            scores = []
            for perc in r["drr_perceptions"]:
                if perc == "liberal":
                    scores.append(1.0)         # active deception (full credit)
                elif perc == "unknown":
                    scores.append(1.0)         # ambiguity (deception success)
                elif perc in ("fascist", "hitler"):
                    actual = r["alice_role"]
                    if perc == actual:
                        scores.append(0.0)     # correct ID
                    else:
                        scores.append(0.5)     # half — wrong evil role
            drr_per_game.append(scores)
        # GSIR — any game with at least 1 alice action
        actions = gsir_actions_for_game(r)
        if actions:
            gsir_per_game.append(actions)
        # RIA — only Alice-liberal games (ALICE_ONLY convention)
        ria = ria_scores_for_game(r)
        if ria:
            ria_per_game.append(ria)
    return dict(
        lib=np.array(lib_wins),
        fas=np.array(fas_wins),
        hit=np.array(hit_wins),
        drr_per_game=drr_per_game,
        gsir_per_game=gsir_per_game,
        ria_per_game=ria_per_game,
    )


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def stratified_winrate_boot(lib, fas, hit, B=B, rng=None) -> np.ndarray:
    """Bootstrap the *overall* win rate under stratified resampling per role.

    Stratification per role preserves the 60/20/20 split, which matches how
    the experiment was actually run (sim.sh forces these proportions). Without
    stratification, a bootstrap on the games as-a-whole would also resample the
    role distribution — and the paper's reported win rate is conditioned on
    that distribution. Stratification gives a CI that quantifies *only* the
    metric uncertainty under the fixed design.
    """
    rng = rng or RNG
    if len(lib) == 0 and len(fas) == 0 and len(hit) == 0:
        return np.array([])
    means = np.empty(B)
    nl, nfas, nh = len(lib), len(fas), len(hit)
    total = nl + nfas + nh
    if total == 0:
        return np.array([])
    for b in range(B):
        l = lib[rng.integers(0, nl, nl)].sum() if nl else 0
        f = fas[rng.integers(0, nfas, nfas)].sum() if nfas else 0
        h = hit[rng.integers(0, nh, nh)].sum() if nh else 0
        means[b] = (l + f + h) / total
    return means


def boot_mean(values: np.ndarray, B=B, rng=None) -> np.ndarray:
    rng = rng or RNG
    n = len(values)
    if n == 0:
        return np.array([])
    idx = rng.integers(0, n, size=(B, n))
    return values[idx].mean(axis=1)


def boot_drr(drr_per_game: list[list[float]], B=B, rng=None) -> np.ndarray:
    """Cluster bootstrap by game: resample games with replacement, then pool
    the perception events within each resampled game. This respects the
    correlation between perceptions of the same model in the same game."""
    rng = rng or RNG
    n = len(drr_per_game)
    if n == 0:
        return np.array([])
    # Pre-convert to numpy arrays for speed.
    arr = [np.asarray(g) for g in drr_per_game]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        num = 0.0
        den = 0
        for j in idx:
            num += arr[j].sum() if arr[j].size else 0.0
            den += arr[j].size
        out[b] = (num / den) if den else np.nan
    return out


def boot_gsir(gsir_per_game: list[list[float]], B=B, rng=None) -> np.ndarray:
    """Cluster bootstrap by game for per-action GSIR (in centiscore units)."""
    rng = rng or RNG
    n = len(gsir_per_game)
    if n == 0:
        return np.array([])
    arr = [np.asarray(g, dtype=float) for g in gsir_per_game]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        num = 0.0
        den = 0
        for j in idx:
            num += arr[j].sum()
            den += arr[j].size
        out[b] = (num / den) if den else np.nan
    return 100.0 * out  # centiscore


def boot_ria(ria_per_game: list[list[float]], B=B, rng=None) -> np.ndarray:
    """Cluster bootstrap by game for RIA (% scale)."""
    rng = rng or RNG
    n = len(ria_per_game)
    if n == 0:
        return np.array([])
    arr = [np.asarray(g, dtype=float) for g in ria_per_game]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        num = 0.0
        den = 0
        for j in idx:
            num += arr[j].sum()
            den += arr[j].size
        out[b] = (num / den) if den else np.nan
    return 100.0 * out  # %


# ---------------------------------------------------------------------------
# Paired bootstrap test
# ---------------------------------------------------------------------------

def paired_winrate_pvalue(a_arrays, b_arrays, B=B, rng=None) -> float:
    """Paired bootstrap p-value for H0: WR(a) == WR(b).

    Pairing is stratified per role: at each iteration, sample one index from
    each role bucket and reuse it for both models. The two models played the
    same 60/20/20 role design, so the pairing eliminates the role-mix variance.
    Uses |observed diff| via two-sided percentile rule.
    """
    rng = rng or RNG
    nl = min(len(a_arrays["lib"]), len(b_arrays["lib"]))
    nf = min(len(a_arrays["fas"]), len(b_arrays["fas"]))
    nh = min(len(a_arrays["hit"]), len(b_arrays["hit"]))
    if nl + nf + nh == 0:
        return float("nan")

    def overall_winrate(arr_a, arr_b, idx_l, idx_f, idx_h):
        s_a = arr_a["lib"][idx_l].sum() + arr_a["fas"][idx_f].sum() + arr_a["hit"][idx_h].sum()
        s_b = arr_b["lib"][idx_l].sum() + arr_b["fas"][idx_f].sum() + arr_b["hit"][idx_h].sum()
        n = nl + nf + nh
        return s_a / n, s_b / n

    # Truncate to common length so pairing is meaningful.
    a = {k: v[:nl] if k == "lib" else (v[:nf] if k == "fas" else v[:nh]) for k, v in a_arrays.items() if k in ("lib", "fas", "hit")}
    b = {k: v[:nl] if k == "lib" else (v[:nf] if k == "fas" else v[:nh]) for k, v in b_arrays.items() if k in ("lib", "fas", "hit")}

    wa = (a["lib"].sum() + a["fas"].sum() + a["hit"].sum()) / (nl + nf + nh)
    wb = (b["lib"].sum() + b["fas"].sum() + b["hit"].sum()) / (nl + nf + nh)
    obs_diff = wa - wb

    diffs = np.empty(B)
    for i in range(B):
        idx_l = rng.integers(0, nl, nl) if nl else np.array([], dtype=int)
        idx_f = rng.integers(0, nf, nf) if nf else np.array([], dtype=int)
        idx_h = rng.integers(0, nh, nh) if nh else np.array([], dtype=int)
        s_a, s_b = overall_winrate(a, b, idx_l, idx_f, idx_h)
        diffs[i] = s_a - s_b
    # Centered for null
    centered = diffs - obs_diff
    p = (np.abs(centered) >= np.abs(obs_diff)).mean()
    # Avoid p=0 with finite B.
    return max(p, 1.0 / B)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def ci(boot_samples: np.ndarray) -> tuple[float, float, float]:
    """Backwards-compat: (mean, lo, hi). New code should use cell_stats()."""
    if boot_samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(boot_samples)), float(np.percentile(boot_samples, 2.5)), float(np.percentile(boot_samples, 97.5))


def cell_stats(point: float, boot_samples: np.ndarray) -> dict:
    """Per-cell bundle: point estimate, bootstrap mean, bootstrap SD, 95% CI.

    The bootstrap SD is the empirical standard deviation of the resampled
    means — i.e., a non-parametric estimate of the metric's standard error
    (the reported "standard deviation"). For nominally Gaussian
    sample sizes this is roughly equivalent to a t-based SE; bootstrap
    handles bounded metrics (WR in [0,1], RIA in [0,1]) without the
    Gaussian assumption.
    """
    if boot_samples.size == 0:
        return dict(point=point, boot_mean=float("nan"),
                    sd=float("nan"), ci=[float("nan"), float("nan")])
    return dict(
        point=point,
        boot_mean=float(np.mean(boot_samples)),
        sd=float(np.std(boot_samples, ddof=1)),
        ci=[float(np.percentile(boot_samples, 2.5)),
            float(np.percentile(boot_samples, 97.5))],
    )


def main():
    all_recs = model_records(include_baselines=True)
    arrays = {name: model_arrays(recs) for name, recs in all_recs.items()}

    results = {}
    for name, a in arrays.items():
        n_lib, n_fas, n_hit = len(a["lib"]), len(a["fas"]), len(a["hit"])
        wr_overall = (a["lib"].sum() + a["fas"].sum() + a["hit"].sum()) / max(1, n_lib + n_fas + n_hit)
        wr_boot   = stratified_winrate_boot(a["lib"], a["fas"], a["hit"]) * 100
        wr_lib_boot = boot_mean(a["lib"]) * 100
        wr_fas_boot = boot_mean(a["fas"]) * 100
        wr_hit_boot = boot_mean(a["hit"]) * 100
        drr_boot  = boot_drr(a["drr_per_game"]) * 100
        gsir_boot = boot_gsir(a["gsir_per_game"])
        ria_boot  = boot_ria(a["ria_per_game"])

        drr_n = sum(len(g) for g in a["drr_per_game"])
        drr_point = float(sum(sum(g) for g in a["drr_per_game"]) / max(1, drr_n)) * 100
        gsir_n = sum(len(g) for g in a["gsir_per_game"])
        gsir_point = float(sum(sum(g) for g in a["gsir_per_game"]) / max(1, gsir_n)) * 100
        ria_n = sum(len(g) for g in a["ria_per_game"])
        ria_point = float(sum(sum(g) for g in a["ria_per_game"]) / max(1, ria_n)) * 100 if ria_n else float("nan")

        results[name] = dict(
            n_games=n_lib + n_fas + n_hit,
            n_lib=n_lib, n_fas=n_fas, n_hit=n_hit,
            win_rate=         cell_stats(float(wr_overall) * 100, wr_boot),
            win_rate_liberal= cell_stats(float(a["lib"].mean()) * 100 if n_lib else float("nan"), wr_lib_boot),
            win_rate_fascist= cell_stats(float(a["fas"].mean()) * 100 if n_fas else float("nan"), wr_fas_boot),
            win_rate_hitler=  cell_stats(float(a["hit"].mean()) * 100 if n_hit else float("nan"), wr_hit_boot),
            drr=              dict(n_perceptions=int(drr_n),  **cell_stats(drr_point, drr_boot)),
            gsir=             dict(n_actions=int(gsir_n),     **cell_stats(gsir_point, gsir_boot)),
            ria=              dict(n_perceptions=int(ria_n),  **cell_stats(ria_point, ria_boot)),
        )

    # ---- Paired test #1 — every model vs the next-ranked (existing) ----
    by_wr = sorted(results.items(), key=lambda kv: -kv[1]["win_rate"]["point"])
    pairs_next = []
    for i, (name, _) in enumerate(by_wr[:-1]):
        other = by_wr[i + 1][0]
        p = paired_winrate_pvalue(arrays[name], arrays[other])
        pairs_next.append(dict(model=name, vs=other, p_paired_winrate=float(p)))

    # ---- Paired test #2 — every model vs the GLOBAL #2 model ----
    runnerup = by_wr[1][0]
    pairs_runnerup = []
    for name, _ in by_wr:
        if name == runnerup:
            pairs_runnerup.append(dict(model=name, vs=runnerup, p_paired_winrate=float("nan")))
            continue
        p = paired_winrate_pvalue(arrays[name], arrays[runnerup])
        pairs_runnerup.append(dict(model=name, vs=runnerup, p_paired_winrate=float(p)))

    # ---- Paired test #3 — 5×5 head-to-head matrix on the top 5 (existing) ----
    top5 = [n for n, _ in by_wr[:5]]
    head_matrix = {}
    for a in top5:
        head_matrix[a] = {}
        for b in top5:
            if a == b:
                head_matrix[a][b] = None
                continue
            head_matrix[a][b] = float(paired_winrate_pvalue(arrays[a], arrays[b]))

    out = dict(
        per_model=results,
        ranked_paired_vs_next=pairs_next,
        paired_vs_runnerup=pairs_runnerup,
        runnerup_name=runnerup,
        top5_paired_matrix=head_matrix,
        B=B,
        notes=(
            "Win-rate CIs use a stratified bootstrap by role (60/20/20). "
            "DRR / GSIR / RIA use a cluster bootstrap by game. Each cell "
            "carries: point estimate, bootstrap mean, bootstrap SD (the "
            "standard-deviation analogue), and 95% CI. "
            "Paired tests are reported in three flavours: vs the next-"
            "ranked model, vs the global runner-up (the 'second-best'), "
            "and the full top-5 head-to-head matrix. Within-experiment "
            "uncertainty only — five independent seeds per cell would "
            "require 7000 additional games and is NOT included."
        ),
    )
    out_path = RESULTS_DIR / "bootstrap_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

    # Pretty-print: point ± SD  [CI]
    def fmt(d):
        p, sd = d["point"], d.get("sd", float("nan"))
        lo, hi = d["ci"][0], d["ci"][1]
        if math.isnan(p):
            return "         n/a         "
        return f"{p:5.1f}±{sd:4.1f}[{lo:4.1f},{hi:5.1f}]"

    print("\n" + "=" * 145)
    print(f"{'Model':28s} {'WR % (overall)':>22s} {'Liberal %':>22s} {'Fascist %':>22s} {'Hitler %':>22s}")
    print("=" * 145)
    for name, _ in by_wr:
        r = results[name]
        print(f"{name:28s} {fmt(r['win_rate']):>22s} {fmt(r['win_rate_liberal']):>22s} "
              f"{fmt(r['win_rate_fascist']):>22s} {fmt(r['win_rate_hitler']):>22s}")
    print("\n" + "=" * 100)
    print(f"{'Model':28s} {'DRR %':>22s} {'GSIR (cs/action)':>22s} {'RIA %':>22s}")
    print("=" * 100)
    for name, _ in by_wr:
        r = results[name]
        print(f"{name:28s} {fmt(r['drr']):>22s} {fmt(r['gsir']):>22s} {fmt(r['ria']):>22s}")

    print(f"\nPaired bootstrap p-values vs next-ranked (overall WR):")
    for p in pairs_next:
        print(f"  {p['model']:28s} vs {p['vs']:28s}  p = {p['p_paired_winrate']:.4f}")
    print(f"\nPaired bootstrap p-values vs runner-up ({runnerup}) ['second-best']:")
    for p in pairs_runnerup:
        if p["model"] == runnerup:
            continue
        print(f"  {p['model']:28s} vs {p['vs']:28s}  p = {p['p_paired_winrate']:.4f}")


if __name__ == "__main__":
    main()
