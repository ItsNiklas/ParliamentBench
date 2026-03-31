"""Loaded-terms vs neutral-theme comparison, per arm.

For each (model, condition) pair, computes:
  - n games (Alice = Player 0, ALICE_ID convention)
  - overall Alice win rate + Wilson 95% CI
  - Alice win rate per role (liberal / fascist / hitler) + Wilson CI
  - win-condition distribution (liberal_policies / fascist_policies /
    hitler_chancellor / hitler_killed)
  - mean game length (rounds)
  - empty-chat rate (proxy for refusal — Alice's chat messages that are
    empty strings or None)

For each arm we also emit two-proportion z-tests on overall win rate and
on each per-role win rate (loaded vs neutral).

Outputs JSON to analysis/results/neutral_theme_comparison.json and prints a
markdown-ready summary.

Usage:
    python -m analysis.neutral_theme_comparison
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

ALICE_ID = 0
# Root directory holding the recorded game-run folders. Override with RUNS_ROOT.
# Both conditions of each arm are read from the same root; the loaded arm reuses
# the main-experiment run folder, the neutral arm its `-NEUTRAL` counterpart
# produced by `parliamentbench --theme neutral`.
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", Path(__file__).resolve().parents[1]))
# Where this analysis writes its JSON output.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ARMS = {
    "gemma":   {"label": "Gemma 3 27B",
                "loaded":  RUNS_ROOT / "runsF2-GEMMA",
                "neutral": RUNS_ROOT / "runsF2-GEMMA-NEUTRAL"},
    "mistral": {"label": "Mistral Small 24B",
                "loaded":  RUNS_ROOT / "runsF2-MISTRALSMALL",
                "neutral": RUNS_ROOT / "runsF2-MISTRALSMALL-NEUTRAL"},
    "gptoss":  {"label": "GPT-OSS 120B",
                "loaded":  RUNS_ROOT / "runsF2-GPTOSS120B",
                "neutral": RUNS_ROOT / "runsF2-GPTOSS120B-NEUTRAL"},
}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(k1, n1, k2, n2):
    """Two-proportion z-test, two-sided. Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # two-sided p via erf
    pval = math.erfc(abs(z) / math.sqrt(2))
    return (z, pval)


def parse_game(d):
    players = d.get("players", [])
    logs = d.get("logs", [])
    chats = d.get("chats", [])
    hitler_id = next((i for i, p in enumerate(players) if p.get("role") == "hitler"), None)
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

    role = players[ALICE_ID].get("role") if len(players) > ALICE_ID else None
    won = (role == "liberal" and winner == "liberals") or (
        role in ("fascist", "hitler") and winner == "fascists"
    )

    alice_chats = [c for c in chats if isinstance(c, dict) and c.get("userName") == "Alice"]
    n_alice_chats = len(alice_chats)
    n_empty = sum(1 for c in alice_chats if not (c.get("chat") or "").strip())
    # Refusal heuristic — Alice utterances that trigger any refusal cue.
    REFUSAL_CUES = (
        "as an ai", "i cannot", "i can't comply", "i can not", "i'm sorry, but i",
        "i am not able to", "i'm unable to", "i refuse",
        "i won't engage", "i will not engage",
        "against my guidelines", "against my values",
        "i don't feel comfortable", "i do not feel comfortable",
    )
    n_refusal = 0
    for c in alice_chats:
        txt = (c.get("chat") or "").lower()
        if any(cue in txt for cue in REFUSAL_CUES):
            n_refusal += 1

    return {
        "role": role,
        "won": won,
        "win_condition": cond,
        "winner": winner,
        "rounds": len(logs),
        "n_alice_chats": n_alice_chats,
        "n_empty_chats": n_empty,
        "n_refusal_chats": n_refusal,
    }


def load_dir(p: Path):
    out = []
    for fp in sorted(p.glob("*_summary.json")):
        with open(fp) as f:
            d = json.load(f)
        out.append(parse_game(d))
    return out


def summarize(games):
    n = len(games)
    if n == 0:
        return {"n": 0}
    overall_wins = sum(1 for g in games if g["won"])
    rolewise = {}
    for role in ("liberal", "fascist", "hitler"):
        sub = [g for g in games if g["role"] == role]
        k = sum(1 for g in sub if g["won"])
        lo, hi = wilson(k, len(sub))
        rolewise[role] = {"n": len(sub), "wins": k,
                          "win_rate": (k / len(sub)) if sub else 0.0,
                          "ci": [lo, hi]}
    olo, ohi = wilson(overall_wins, n)
    wc_counts = {"liberal_policies": 0, "fascist_policies": 0,
                 "hitler_chancellor": 0, "hitler_killed": 0}
    for g in games:
        wc_counts[g["win_condition"]] += 1
    avg_rounds = sum(g["rounds"] for g in games) / n
    tot_chats = sum(g["n_alice_chats"] for g in games)
    tot_empty = sum(g["n_empty_chats"] for g in games)
    tot_refusal = sum(g["n_refusal_chats"] for g in games)
    empty_rate = (tot_empty / tot_chats) if tot_chats else 0.0
    refusal_rate = (tot_refusal / tot_chats) if tot_chats else 0.0
    return {
        "n": n,
        "overall_wins": overall_wins,
        "win_rate": overall_wins / n,
        "win_rate_ci": [olo, ohi],
        "by_role": rolewise,
        "win_conditions": wc_counts,
        "avg_rounds": avg_rounds,
        "alice_chat_count": tot_chats,
        "alice_empty_chats": tot_empty,
        "empty_chat_rate": empty_rate,
        "alice_refusal_chats": tot_refusal,
        "refusal_chat_rate": refusal_rate,
    }


def compare(loaded, neutral):
    z, p = two_proportion_z(loaded["overall_wins"], loaded["n"],
                             neutral["overall_wins"], neutral["n"])
    role_tests = {}
    for role in ("liberal", "fascist", "hitler"):
        L, N = loaded["by_role"][role], neutral["by_role"][role]
        z_r, p_r = two_proportion_z(L["wins"], L["n"], N["wins"], N["n"])
        role_tests[role] = {
            "loaded_wr": L["win_rate"], "neutral_wr": N["win_rate"],
            "delta": N["win_rate"] - L["win_rate"],
            "z": z_r, "p": p_r,
            "n_loaded": L["n"], "n_neutral": N["n"],
        }
    return {
        "overall": {
            "loaded_wr": loaded["win_rate"], "neutral_wr": neutral["win_rate"],
            "delta": neutral["win_rate"] - loaded["win_rate"],
            "z": z, "p": p,
            "n_loaded": loaded["n"], "n_neutral": neutral["n"],
        },
        "by_role": role_tests,
    }


def main():
    out = {"arms": {}}
    for key, info in ARMS.items():
        loaded_games = load_dir(info["loaded"])
        neutral_games = load_dir(info["neutral"])
        loaded = summarize(loaded_games)
        neutral = summarize(neutral_games)
        out["arms"][key] = {
            "label": info["label"],
            # Folder names only — never absolute local paths.
            "paths": {"loaded": info["loaded"].name,
                       "neutral": info["neutral"].name},
            "loaded": loaded,
            "neutral": neutral,
            "compare": compare(loaded, neutral),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "neutral_theme_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # Markdown-ready printout
    for key, arm in out["arms"].items():
        L, N = arm["loaded"], arm["neutral"]
        C = arm["compare"]
        print(f"\n## {arm['label']}  (loaded n={L['n']}, neutral n={N['n']})")
        print(f"  Overall WR: loaded {L['win_rate']*100:.1f}% [{L['win_rate_ci'][0]*100:.1f},{L['win_rate_ci'][1]*100:.1f}]  "
              f"neutral {N['win_rate']*100:.1f}% [{N['win_rate_ci'][0]*100:.1f},{N['win_rate_ci'][1]*100:.1f}]  "
              f"Δ={C['overall']['delta']*100:+.1f}pp  z={C['overall']['z']:+.2f}  p={C['overall']['p']:.3f}")
        for role in ("liberal", "fascist", "hitler"):
            rL, rN = L["by_role"][role], N["by_role"][role]
            rT = C["by_role"][role]
            print(f"  {role:8s}: loaded {rL['win_rate']*100:5.1f}% ({rL['wins']}/{rL['n']}) "
                  f"[{rL['ci'][0]*100:4.1f},{rL['ci'][1]*100:5.1f}]  "
                  f"neutral {rN['win_rate']*100:5.1f}% ({rN['wins']}/{rN['n']}) "
                  f"[{rN['ci'][0]*100:4.1f},{rN['ci'][1]*100:5.1f}]  "
                  f"Δ={rT['delta']*100:+5.1f}pp  z={rT['z']:+.2f}  p={rT['p']:.3f}")
        print(f"  Win conditions (loaded → neutral):")
        for wc in ("liberal_policies", "fascist_policies", "hitler_chancellor", "hitler_killed"):
            print(f"    {wc:20s}: {L['win_conditions'][wc]:3d} → {N['win_conditions'][wc]:3d}")
        print(f"  Avg rounds: loaded {L['avg_rounds']:.2f}, neutral {N['avg_rounds']:.2f}")
        print(f"  Alice chat count: loaded {L['alice_chat_count']}, neutral {N['alice_chat_count']}")
        print(f"  Alice empty-chat rate: loaded {L['empty_chat_rate']*100:.2f}% ({L['alice_empty_chats']}/{L['alice_chat_count']}), "
              f"neutral {N['empty_chat_rate']*100:.2f}% ({N['alice_empty_chats']}/{N['alice_chat_count']})")
        print(f"  Alice refusal-chat rate: loaded {L['refusal_chat_rate']*100:.2f}% ({L['alice_refusal_chats']}/{L['alice_chat_count']}), "
              f"neutral {N['refusal_chat_rate']*100:.2f}% ({N['alice_refusal_chats']}/{N['alice_chat_count']})")


if __name__ == "__main__":
    main()
