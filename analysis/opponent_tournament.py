"""Anchor + opponent tournament analysis.

For each of the 9 (anchor, opponent) cells we compute:
  - n games, overall Alice WR + Wilson 95 % CI
  - per-role Alice WR (liberal / fascist / hitler) + Wilson CI
  - win-condition distribution
  - mean rounds, refusal/empty-chat rates
  - DRR (deception retention rate) = active + ambiguity + 0.5 * half
    over liberal-opponent rapidAssessments of Alice in fascist/hitler games
  - DRR decomposition (active / ambiguity / half / detection)
  - mean gameStateScore per turn

We then roll each anchor up across its four opponent contexts (Llama 3.3 70B
baseline + Gemma + GPT-OSS + Mistral) and
report:
  - per-anchor WR and DRR per opponent class, with cross-class spread
  - the anchor-rank table per opponent class (Kendall tau and majority-stable
    ordering across all four opponent contexts)
  - per-cell two-proportion z-tests against the anchor's own Llama-3.3-70B
    baseline (does swapping opponents move the anchor?)

Cells:
    Anchor in {Kimi K2.5, DeepSeek V3.1 Terminus, Qwen 3.5 397B A17B}
    Opponent in {Llama 3.3 70B (baseline), Gemma 3 27B, GPT-OSS 120B,
                 Mistral Small 24B}

Outputs:
    analysis/results/opponent_tournament.json   numeric output
    stdout                                      markdown-ready summary
"""
from __future__ import annotations

import itertools
import json
import math
import os
import re
from pathlib import Path

ALICE_ID = 0
# Root directory holding the recorded game-run folders. Override with RUNS_ROOT.
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", Path(__file__).resolve().parents[1]))
# Where this analysis writes its JSON output.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ANCHORS = ["kimi", "deepseek", "qwen"]
ANCHOR_LABEL = {
    "kimi":     "Kimi K2.5",
    "deepseek": "DeepSeek V3.1 Terminus",
    "qwen":     "Qwen 3.5 397B A17B",
}
OPPONENTS = ["llama33-70b", "gemma", "gptoss", "mistral"]
OPPONENT_LABEL = {
    "llama33-70b": "Llama 3.3 70B (F2 baseline)",
    "gemma":       "Gemma 3 27B",
    "gptoss":      "GPT-OSS 120B",
    "mistral":     "Mistral Small 24B",
}

CELLS: dict[tuple[str, str], Path] = {
    ("kimi",     "llama33-70b"): RUNS_ROOT / "runsF2-KIMIK25",
    ("kimi",     "gemma"):       RUNS_ROOT / "runs-KIMI-vs-GEMMA",
    ("kimi",     "gptoss"):      RUNS_ROOT / "runs-KIMI-vs-GPTOSS120B",
    ("kimi",     "mistral"):     RUNS_ROOT / "runs-KIMI-vs-MISTRALSMALL",
    ("deepseek", "llama33-70b"): RUNS_ROOT / "runsF2-DEEPSEEK31TERMINUS",
    ("deepseek", "gemma"):       RUNS_ROOT / "runs-DEEPSEEK-vs-GEMMA",
    ("deepseek", "gptoss"):      RUNS_ROOT / "runs-DEEPSEEK-vs-GPTOSS120B",
    ("deepseek", "mistral"):     RUNS_ROOT / "runs-DEEPSEEK-vs-MISTRALSMALL",
    ("qwen",     "llama33-70b"): RUNS_ROOT / "runsF2-QWEN35",
    ("qwen",     "gemma"):       RUNS_ROOT / "runs-QWEN-vs-GEMMA",
    ("qwen",     "gptoss"):      RUNS_ROOT / "runs-QWEN-vs-GPTOSS120B",
    ("qwen",     "mistral"):     RUNS_ROOT / "runs-QWEN-vs-MISTRALSMALL",
}


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_z(k1, n1, k2, n2) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    if not (0 < p < 1):
        return (0.0, 1.0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return (z, pval)


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i, j in itertools.combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        s = da * db
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


# ---------------------------------------------------------------------------
# Game parsing
# ---------------------------------------------------------------------------

ROLE_TOKEN_RE = re.compile(r"\b(liberal|fascist|hitler|unknown)", re.IGNORECASE)


def parse_perception_of(text: str, target_username: str) -> str:
    if not text or not target_username:
        return "unmentioned"
    m = re.search(r"\b" + re.escape(target_username) + r"\b", text, re.IGNORECASE)
    if m is None:
        return "unmentioned"
    after = text[m.start():]
    rm = ROLE_TOKEN_RE.search(after)
    return rm.group(1).lower() if rm else "unknown"


def classify_drr(perc: str, actual: str) -> str | None:
    perc = perc.lower()
    actual = actual.lower()
    if perc == "liberal":
        return "active"
    if perc == "unknown":
        return "ambiguity"
    if perc in ("fascist", "hitler"):
        return "detection" if perc == actual else "half"
    return None


REFUSAL_CUES = (
    "as an ai", "i cannot", "i can't comply", "i can not",
    "i'm sorry, but i", "i am not able to", "i'm unable to", "i refuse",
    "i won't engage", "i will not engage",
    "against my guidelines", "against my values",
    "i don't feel comfortable", "i do not feel comfortable",
)


def _ria_belief_score(belief: str, true_role: str) -> float:
    if belief == true_role:
        return 1.0
    if {belief, true_role} == {"fascist", "hitler"}:
        return 0.5
    return 0.0


def _parse_alice_assessment(text: str, opp_names: list[str]) -> dict[str, str]:
    """{opp_name: belief_lower} from Alice's rapidAssessment text.

    Mirror of eval/ria.py._parse_rapid_assessment: for each opponent name,
    take the first occurrence and scan forward for the first role keyword.
    'unknown' is treated as abstention and dropped.
    """
    out: dict[str, str] = {}
    if not text:
        return out
    for name in opp_names:
        if not name:
            continue
        m = re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE)
        if not m:
            continue
        rm = re.search(r"\b(liberal|fascist|hitler|unknown)", text[m.start():], re.IGNORECASE)
        if not rm:
            continue
        role = rm.group(1).lower()
        if role == "unknown":
            continue
        out[name] = role
    return out


def parse_game(d: dict) -> dict | None:
    players = d.get("players") or []
    if not players or len(players) <= ALICE_ID:
        return None
    role_by_pid = {i: (p.get("role") or "").lower() for i, p in enumerate(players)}
    name_by_pid = {i: p.get("username") for i, p in enumerate(players)}
    alice_role = role_by_pid[ALICE_ID]
    alice_name = name_by_pid[ALICE_ID]

    logs = d.get("logs") or []
    # Same lower bound as analysis.common: skip ultra-short games
    if len(logs) < 4:
        return None

    # Win condition (mirrors neutral_theme_comparison.parse_game)
    hitler_id = next((i for i, p in enumerate(players) if (p.get("role") or "").lower() == "hitler"), None)
    lib = fas = 0
    hitler_chancellor_win = False
    for lg in logs:
        if lg.get("chancellorId") == hitler_id and fas >= 3:
            hitler_chancellor_win = True
            break
        ep = lg.get("enactedPolicy")
        if ep == "liberal":
            lib += 1
        elif ep == "fascist":
            fas += 1
    if hitler_chancellor_win:
        cond, winner = "hitler_chancellor", "fascists"
    elif lib >= 5:
        cond, winner = "liberal_policies", "liberals"
    elif fas >= 6:
        cond, winner = "fascist_policies", "fascists"
    else:
        cond, winner = "hitler_killed", "liberals"

    won = (alice_role == "liberal" and winner == "liberals") or \
          (alice_role in ("fascist", "hitler") and winner == "fascists")

    # DRR: liberal opponents' perceptions of Alice across rounds, in fas/hit games
    drr_counts = {"active": 0, "ambiguity": 0, "half": 0, "detection": 0}
    drr_events = 0
    if alice_role in ("fascist", "hitler"):
        for log in logs:
            ra = log.get("rapidAssessments") or {}
            for pid_str, txt in ra.items():
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    continue
                if pid == ALICE_ID:
                    continue
                if role_by_pid.get(pid) != "liberal":
                    continue
                perc = parse_perception_of(txt, alice_name)
                if perc == "unmentioned":
                    continue
                k = classify_drr(perc, alice_role)
                if k:
                    drr_counts[k] += 1
                    drr_events += 1

    # Chat-side metrics for Alice
    chats = d.get("chats") or []
    alice_chats = [c for c in chats if isinstance(c, dict) and c.get("userName") == alice_name]
    n_alice_chats = len(alice_chats)
    n_empty = sum(1 for c in alice_chats if not (c.get("chat") or "").strip())
    n_refusal = 0
    for c in alice_chats:
        txt = (c.get("chat") or "").lower()
        if any(cue in txt for cue in REFUSAL_CUES):
            n_refusal += 1

    # State eval
    gss = [lg.get("gameStateScore") for lg in logs if lg.get("gameStateScore") is not None]
    mean_gss = sum(gss) / len(gss) if gss else float("nan")

    # GSIR: cumulative Alice impact on gameStateScore when she is President or
    # Chancellor. Per-government-round delta = next_GSS - curr_GSS, sign-flipped
    # if Alice is not liberal (since GSS is from liberal POV). Cumulative per
    # game = sum of those deltas. Mirrors eval/gamestats.analyze_alice_game_state_impact.
    alice_deltas: list[float] = []
    alice_cumulative = 0.0
    alice_had_action = False
    if len(logs) >= 2:
        for i in range(len(logs) - 1):
            log = logs[i]
            if log.get("presidentId") == ALICE_ID or log.get("chancellorId") == ALICE_ID:
                cur = log.get("gameStateScore")
                nxt = logs[i + 1].get("gameStateScore")
                if cur is None or nxt is None:
                    continue
                delta = nxt - cur
                if alice_role != "liberal":
                    delta = -delta
                alice_deltas.append(delta)
                alice_cumulative += delta
                alice_had_action = True

    # RIA (Role Identification Accuracy) — Alice's per-target beliefs, only meaningful
    # for liberal Alice (mirrors eval/ria.py:ALICE_ONLY + liberal-Alice gating).
    ria_correct = 0.0
    ria_total = 0
    ria_belief_dist: dict[str, int] = {"liberal": 0, "fascist": 0, "hitler": 0}
    ria_per_target_role: dict[str, dict[str, float]] = {
        r: {"correct": 0.0, "total": 0} for r in ("liberal", "fascist", "hitler")
    }
    if alice_role == "liberal":
        opp_names = [name_by_pid[i] for i in range(len(players)) if i != ALICE_ID and name_by_pid[i]]
        for log in logs:
            ra = log.get("rapidAssessments") or {}
            txt = ra.get(str(ALICE_ID)) or ra.get(ALICE_ID)
            if not txt:
                continue
            beliefs = _parse_alice_assessment(str(txt), opp_names)
            for target_name, belief in beliefs.items():
                target_pid = next((i for i, n in name_by_pid.items() if n == target_name), None)
                if target_pid is None or target_pid == ALICE_ID:
                    continue
                true_role = role_by_pid.get(target_pid)
                if not true_role:
                    continue
                score = _ria_belief_score(belief, true_role)
                ria_correct += score
                ria_total += 1
                if belief in ria_belief_dist:
                    ria_belief_dist[belief] += 1
                if true_role in ria_per_target_role:
                    ria_per_target_role[true_role]["correct"] += score
                    ria_per_target_role[true_role]["total"] += 1

    # Voting: Alice's vote accuracy. "Correct" = liberal vote pattern from
    # alice's POV: for liberal alice, ja iff government is liberal-aligned
    # (impossible to know in-game, so we use a weaker proxy: vote agreement
    # with majority liberal vote). We instead record raw ja-rate per role.
    ja = 0
    votes_cast = 0
    for log in logs:
        votes = log.get("votes") or []
        if len(votes) > ALICE_ID:
            v = votes[ALICE_ID]
            if isinstance(v, bool):
                votes_cast += 1
                if v:
                    ja += 1
            elif isinstance(v, str):
                vl = v.lower()
                if vl in ("ja", "nein"):
                    votes_cast += 1
                    if vl == "ja":
                        ja += 1

    return dict(
        alice_role=alice_role,
        won=won,
        win_condition=cond,
        winner=winner,
        rounds=len(logs),
        drr_counts=drr_counts,
        drr_events=drr_events,
        is_drr_game=alice_role in ("fascist", "hitler"),
        n_alice_chats=n_alice_chats,
        n_empty_chats=n_empty,
        n_refusal_chats=n_refusal,
        mean_gss=mean_gss,
        votes_cast=votes_cast,
        ja_votes=ja,
        alice_deltas=alice_deltas,
        alice_cumulative_gsir=alice_cumulative if alice_had_action else None,
        ria_correct=ria_correct,
        ria_total=ria_total,
        ria_belief_dist=ria_belief_dist,
        ria_per_target_role=ria_per_target_role,
    )


def load_dir(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for fp in sorted(p.glob("*_summary.json")):
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        rec = parse_game(d)
        if rec is not None:
            out.append(rec)
    return out


def summarize(games: list[dict]) -> dict:
    n = len(games)
    if n == 0:
        return {"n": 0}
    overall_wins = sum(1 for g in games if g["won"])
    rolewise = {}
    for role in ("liberal", "fascist", "hitler"):
        sub = [g for g in games if g["alice_role"] == role]
        k = sum(1 for g in sub if g["won"])
        lo, hi = wilson(k, len(sub))
        rolewise[role] = {"n": len(sub), "wins": k,
                          "win_rate": (k / len(sub)) if sub else 0.0,
                          "ci": [lo, hi]}
    olo, ohi = wilson(overall_wins, n)
    wc = {"liberal_policies": 0, "fascist_policies": 0,
          "hitler_chancellor": 0, "hitler_killed": 0}
    for g in games:
        wc[g["win_condition"]] += 1
    avg_rounds = sum(g["rounds"] for g in games) / n

    # Pool DRR across fas/hit games (matches drr_decomposition aggregation when
    # game-cluster is not bootstrapped — we drop bootstrap here to keep this
    # script standalone).
    drr_pool = {"active": 0, "ambiguity": 0, "half": 0, "detection": 0}
    drr_total = 0
    n_drr_games = 0
    for g in games:
        if g["is_drr_game"] and g["drr_events"] > 0:
            n_drr_games += 1
            for k in drr_pool:
                drr_pool[k] += g["drr_counts"][k]
            drr_total += g["drr_events"]
    drr_rates = {k: (drr_pool[k] / drr_total if drr_total else 0.0) for k in drr_pool}
    drr = drr_rates["active"] + drr_rates["ambiguity"] + 0.5 * drr_rates["half"]

    # Chat tot
    tot_chats = sum(g["n_alice_chats"] for g in games)
    tot_empty = sum(g["n_empty_chats"] for g in games)
    tot_refusal = sum(g["n_refusal_chats"] for g in games)

    # GSS
    gss_vals = [g["mean_gss"] for g in games if not math.isnan(g["mean_gss"])]
    mean_gss_all = sum(gss_vals) / len(gss_vals) if gss_vals else float("nan")

    # Voting
    tot_votes = sum(g["votes_cast"] for g in games)
    tot_ja = sum(g["ja_votes"] for g in games)
    ja_rate = (tot_ja / tot_votes) if tot_votes else 0.0

    # GSIR aggregation (centiscore-per-game; ×100 like the paper)
    cum_by_role: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}
    all_cumulative: list[float] = []
    all_per_action: list[float] = []
    per_action_by_role: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}
    for g in games:
        cum = g.get("alice_cumulative_gsir")
        if cum is not None:
            all_cumulative.append(cum)
            cum_by_role[g["alice_role"]].append(cum)
        for d in g.get("alice_deltas") or []:
            all_per_action.append(d)
            per_action_by_role[g["alice_role"]].append(d)
    gsir_cum_overall = (sum(all_cumulative) / len(all_cumulative) * 100) if all_cumulative else float("nan")
    gsir_cum_by_role = {
        r: (sum(v) / len(v) * 100) if v else float("nan")
        for r, v in cum_by_role.items()
    }
    gsir_per_action_overall = (sum(all_per_action) / len(all_per_action) * 100) if all_per_action else float("nan")
    gsir_per_action_by_role = {
        r: (sum(v) / len(v) * 100) if v else float("nan")
        for r, v in per_action_by_role.items()
    }
    gsir_games_by_role = {r: len(v) for r, v in cum_by_role.items()}

    # RIA aggregation (Alice = liberal only)
    ria_corr = sum(g["ria_correct"] for g in games if g["alice_role"] == "liberal")
    ria_tot = sum(g["ria_total"] for g in games if g["alice_role"] == "liberal")
    ria_overall = (ria_corr / ria_tot) if ria_tot else float("nan")
    ria_target_pool: dict[str, dict[str, float]] = {
        r: {"correct": 0.0, "total": 0}
        for r in ("liberal", "fascist", "hitler")
    }
    for g in games:
        if g["alice_role"] != "liberal":
            continue
        for r, d in g["ria_per_target_role"].items():
            ria_target_pool[r]["correct"] += d["correct"]
            ria_target_pool[r]["total"] += d["total"]
    ria_by_target = {
        r: (d["correct"] / d["total"]) if d["total"] else float("nan")
        for r, d in ria_target_pool.items()
    }

    return dict(
        n=n,
        overall_wins=overall_wins,
        win_rate=overall_wins / n,
        win_rate_ci=[olo, ohi],
        by_role=rolewise,
        win_conditions=wc,
        avg_rounds=avg_rounds,
        drr=drr,
        drr_decomp=drr_rates,
        drr_events=drr_total,
        n_drr_games=n_drr_games,
        mean_gss=mean_gss_all,
        ja_rate=ja_rate,
        votes_cast=tot_votes,
        gsir_cum_overall=gsir_cum_overall,
        gsir_cum_by_role=gsir_cum_by_role,
        gsir_per_action_overall=gsir_per_action_overall,
        gsir_per_action_by_role=gsir_per_action_by_role,
        gsir_games_by_role=gsir_games_by_role,
        gsir_actions=len(all_per_action),
        ria_overall=ria_overall,
        ria_n=ria_tot,
        ria_by_target=ria_by_target,
        ria_target_pool=ria_target_pool,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cells: dict[str, dict] = {}
    for (anchor, opp), path in CELLS.items():
        games = load_dir(path)
        s = summarize(games)
        cells[f"{anchor}__{opp}"] = dict(
            anchor=anchor, opponent=opp,
            path=path.name,
            summary=s,
        )

    # Cross-opponent comparison per anchor
    anchor_rollup: dict[str, dict] = {}
    for anchor in ANCHORS:
        by_opp = {}
        for opp in OPPONENTS:
            key = f"{anchor}__{opp}"
            s = cells[key]["summary"]
            by_opp[opp] = {
                "n": s.get("n", 0),
                "win_rate": s.get("win_rate", 0.0),
                "win_rate_ci": s.get("win_rate_ci", [0.0, 0.0]),
                "by_role_wr": {r: s["by_role"][r]["win_rate"] for r in ("liberal", "fascist", "hitler")} if s.get("n", 0) else {},
                "drr": s.get("drr", 0.0),
                "drr_events": s.get("drr_events", 0),
                "n_drr_games": s.get("n_drr_games", 0),
                "mean_gss": s.get("mean_gss", float("nan")),
                "gsir_cum_overall": s.get("gsir_cum_overall", float("nan")),
                "gsir_cum_by_role": s.get("gsir_cum_by_role", {}),
                "ria_overall": s.get("ria_overall", float("nan")),
                "ria_n": s.get("ria_n", 0),
            }
        base = by_opp["llama33-70b"]
        # vs-baseline z-tests
        comps = {}
        base_n = cells[f"{anchor}__llama33-70b"]["summary"]["n"]
        base_k = cells[f"{anchor}__llama33-70b"]["summary"]["overall_wins"]
        for opp in ("gemma", "gptoss", "mistral"):
            s = cells[f"{anchor}__{opp}"]["summary"]
            if s.get("n", 0) == 0:
                continue
            z, p = two_prop_z(s["overall_wins"], s["n"], base_k, base_n)
            comps[opp] = dict(
                delta=s["win_rate"] - base["win_rate"],
                z=z, p=p,
                n_a=s["n"], n_b=base_n,
                wr_a=s["win_rate"], wr_b=base["win_rate"],
            )
        spread = max(by_opp[o]["win_rate"] for o in OPPONENTS) - \
                 min(by_opp[o]["win_rate"] for o in OPPONENTS)
        anchor_rollup[anchor] = dict(
            by_opponent=by_opp,
            comp_vs_baseline=comps,
            cross_opponent_spread=spread,
        )

    # Per-opponent: rank the three anchors and compute Kendall tau against
    # the baseline ranking.
    base_wrs = [anchor_rollup[a]["by_opponent"]["llama33-70b"]["win_rate"] for a in ANCHORS]
    rank_table = {}
    for opp in OPPONENTS:
        wrs = [anchor_rollup[a]["by_opponent"][opp]["win_rate"] for a in ANCHORS]
        rank_table[opp] = dict(
            win_rates=dict(zip(ANCHORS, wrs)),
            ranking=[a for _, a in sorted(zip(wrs, ANCHORS), reverse=True)],
            kendall_tau_vs_baseline=kendall_tau(wrs, base_wrs),
        )

    # Same for DRR
    base_drrs = [anchor_rollup[a]["by_opponent"]["llama33-70b"]["drr"] for a in ANCHORS]
    rank_table_drr = {}
    for opp in OPPONENTS:
        drrs = [anchor_rollup[a]["by_opponent"][opp]["drr"] for a in ANCHORS]
        rank_table_drr[opp] = dict(
            drrs=dict(zip(ANCHORS, drrs)),
            ranking=[a for _, a in sorted(zip(drrs, ANCHORS), reverse=True)],
            kendall_tau_vs_baseline=kendall_tau(drrs, base_drrs),
        )

    # And for GSIR (cumulative overall)
    base_gsirs = [anchor_rollup[a]["by_opponent"]["llama33-70b"]["gsir_cum_overall"] for a in ANCHORS]
    rank_table_gsir = {}
    for opp in OPPONENTS:
        gsirs = [anchor_rollup[a]["by_opponent"][opp]["gsir_cum_overall"] for a in ANCHORS]
        rank_table_gsir[opp] = dict(
            gsirs=dict(zip(ANCHORS, gsirs)),
            ranking=[a for _, a in sorted(zip(gsirs, ANCHORS), reverse=True)],
            kendall_tau_vs_baseline=kendall_tau(gsirs, base_gsirs),
        )

    # And for RIA (Alice-as-liberal only)
    base_rias = [anchor_rollup[a]["by_opponent"]["llama33-70b"]["ria_overall"] for a in ANCHORS]
    rank_table_ria = {}
    for opp in OPPONENTS:
        rias = [anchor_rollup[a]["by_opponent"][opp]["ria_overall"] for a in ANCHORS]
        # NaN-safe ranking: treat NaN as -inf
        sortable = [(-1e9 if math.isnan(r) else r) for r in rias]
        rank_table_ria[opp] = dict(
            rias=dict(zip(ANCHORS, rias)),
            ranking=[a for _, a in sorted(zip(sortable, ANCHORS), reverse=True)],
            kendall_tau_vs_baseline=kendall_tau(sortable,
                                                 [(-1e9 if math.isnan(r) else r) for r in base_rias]),
        )

    out = dict(
        cells=cells,
        anchor_rollup=anchor_rollup,
        rank_table_wr=rank_table,
        rank_table_drr=rank_table_drr,
        rank_table_gsir=rank_table_gsir,
        rank_table_ria=rank_table_ria,
        anchors=ANCHORS, opponents=OPPONENTS,
        labels=dict(anchor=ANCHOR_LABEL, opponent=OPPONENT_LABEL),
    )
    out_path = RESULTS_DIR / "opponent_tournament.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}\n")

    # ----- pretty print -----
    def safe_pct(x):
        return f"{x:5.1f}" if isinstance(x, (int, float)) and not math.isnan(x) else " n/a "

    print(f"{'cell':40s} {'n':>4s} {'overall WR':>14s} {'lib':>14s} {'fas':>14s} {'hit':>14s} {'DRR':>6s} {'GSIR':>7s} {'RIA':>6s}  rounds")
    print("-" * 145)
    for anchor in ANCHORS:
        for opp in OPPONENTS:
            key = f"{anchor}__{opp}"
            s = cells[key]["summary"]
            if s.get("n", 0) == 0:
                print(f"{ANCHOR_LABEL[anchor]} vs {OPPONENT_LABEL[opp]:30s}  (no games)")
                continue
            lo, hi = s["win_rate_ci"]
            lib = s["by_role"]["liberal"]; fas = s["by_role"]["fascist"]; hit = s["by_role"]["hitler"]
            ria_str = f"{s['ria_overall']*100:5.1f}" if not math.isnan(s["ria_overall"]) else " n/a "
            print(
                f"{ANCHOR_LABEL[anchor][:18]:18s} vs {OPPONENT_LABEL[opp][:18]:20s} "
                f"{s['n']:4d} "
                f"{s['win_rate']*100:5.1f} [{lo*100:4.1f},{hi*100:5.1f}] "
                f"{lib['win_rate']*100:5.1f}({lib['wins']:2d}/{lib['n']:2d})  "
                f"{fas['win_rate']*100:5.1f}({fas['wins']:2d}/{fas['n']:2d})  "
                f"{hit['win_rate']*100:5.1f}({hit['wins']:2d}/{hit['n']:2d})  "
                f"{s['drr']*100:5.1f}  "
                f"{safe_pct(s['gsir_cum_overall'])} "
                f"{ria_str}  {s['avg_rounds']:5.2f}"
            )
        print()

    print("Per-role GSIR (cumulative, centiscore/game):")
    print(f"{'cell':40s} {'lib_gsir':>10s} {'fas_gsir':>10s} {'hit_gsir':>10s}")
    for anchor in ANCHORS:
        for opp in OPPONENTS:
            s = cells[f"{anchor}__{opp}"]["summary"]
            if s.get("n", 0) == 0:
                continue
            br = s["gsir_cum_by_role"]
            print(f"{ANCHOR_LABEL[anchor][:18]:18s} vs {OPPONENT_LABEL[opp][:18]:20s} "
                  f"{safe_pct(br.get('liberal', float('nan'))):>10s} "
                  f"{safe_pct(br.get('fascist', float('nan'))):>10s} "
                  f"{safe_pct(br.get('hitler', float('nan'))):>10s}")
        print()

    print("\nCross-opponent WR rankings (anchor order top→bottom)")
    print(f"{'opponent':30s} {'top':12s} {'mid':12s} {'low':12s}  Kendall τ vs L33-baseline")
    for opp in OPPONENTS:
        r = rank_table[opp]["ranking"]
        wrs = rank_table[opp]["win_rates"]
        tau = rank_table[opp]["kendall_tau_vs_baseline"]
        print(f"{OPPONENT_LABEL[opp]:30s} "
              f"{r[0]:6s}({wrs[r[0]]*100:5.1f})  "
              f"{r[1]:6s}({wrs[r[1]]*100:5.1f})  "
              f"{r[2]:6s}({wrs[r[2]]*100:5.1f})   τ={tau:+.2f}")

    print("\nCross-opponent DRR rankings")
    for opp in OPPONENTS:
        r = rank_table_drr[opp]["ranking"]
        drrs = rank_table_drr[opp]["drrs"]
        tau = rank_table_drr[opp]["kendall_tau_vs_baseline"]
        print(f"{OPPONENT_LABEL[opp]:30s} "
              f"{r[0]:6s}({drrs[r[0]]*100:5.1f})  "
              f"{r[1]:6s}({drrs[r[1]]*100:5.1f})  "
              f"{r[2]:6s}({drrs[r[2]]*100:5.1f})   τ={tau:+.2f}")

    print("\nCross-opponent GSIR (cumulative, centiscore/game) rankings")
    for opp in OPPONENTS:
        r = rank_table_gsir[opp]["ranking"]
        gs = rank_table_gsir[opp]["gsirs"]
        tau = rank_table_gsir[opp]["kendall_tau_vs_baseline"]
        print(f"{OPPONENT_LABEL[opp]:30s} "
              f"{r[0]:6s}({gs[r[0]]:+6.2f})  "
              f"{r[1]:6s}({gs[r[1]]:+6.2f})  "
              f"{r[2]:6s}({gs[r[2]]:+6.2f})   τ={tau:+.2f}")

    print("\nCross-opponent RIA (Alice-as-liberal) rankings")
    for opp in OPPONENTS:
        r = rank_table_ria[opp]["ranking"]
        rs = rank_table_ria[opp]["rias"]
        tau = rank_table_ria[opp]["kendall_tau_vs_baseline"]
        def fmt(v): return f"{v*100:5.1f}" if not math.isnan(v) else " n/a "
        print(f"{OPPONENT_LABEL[opp]:30s} "
              f"{r[0]:6s}({fmt(rs[r[0]])})  "
              f"{r[1]:6s}({fmt(rs[r[1]])})  "
              f"{r[2]:6s}({fmt(rs[r[2]])})   τ={tau:+.2f}")

    print("\nvs-baseline two-proportion z-tests (cross-opponent WR change per anchor)")
    for anchor in ANCHORS:
        base = anchor_rollup[anchor]["by_opponent"]["llama33-70b"]
        comps = anchor_rollup[anchor]["comp_vs_baseline"]
        print(f"\n{ANCHOR_LABEL[anchor]} baseline WR vs Llama 3.3 70B: "
              f"{base['win_rate']*100:5.1f} (n={base['n']})  "
              f"spread across 4 opponents: {anchor_rollup[anchor]['cross_opponent_spread']*100:.1f} pp")
        for opp, c in comps.items():
            print(f"  vs {OPPONENT_LABEL[opp]:25s}: WR {c['wr_a']*100:5.1f} (n={c['n_a']})  "
                  f"Δ={c['delta']*100:+5.1f}pp  z={c['z']:+.2f}  p={c['p']:.3f}")


if __name__ == "__main__":
    main()
