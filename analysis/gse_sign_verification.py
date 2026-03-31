"""GSE/GSIR sign-convention verification.

Motivation: there is an apparent disconnect between the reported GSIR in
Table 1 (which appears to show deleterious effects on team when fascist)
and the superior performance in fascist roles. One would have to conclude
that the Llama 3.3 fascist partner has a strong positive GSIR or that
Llama 3.3 liberals always have a very negative GSIR. This script
determines whether there is a bug in the reported GSIR (e.g. the GSE not
being properly negated).

This script traces GSIR for N fascist-winning games (default 5) from a
given run directory. For each game we print, per turn:

  * round, president-id, chancellor-id, enacted policy
  * liberal-perspective gameStateScore (as logged; positive = good for libs)
  * Alice's team-relative gameStateScore (= +score if Alice liberal; -score
    if Alice fascist or hitler)  ← this is the sign convention used by
    eval/gamestats.py:415 and analysis/bootstrap_ci.py
  * the per-turn delta (next − this) of the team-relative score, with
    an "*" marker on turns where Alice was president or chancellor
    (i.e., contributed to GSIR)

At the end of each game we print: Alice's role, the game's winner, the
sum of Alice's *team-relative* action deltas (Alice GSIR_total), and the
sign of that sum.

Expected on a fascist-winning game where Alice is fascist/hitler:
    GSIR_total should be POSITIVE (Alice took actions that moved the
    game toward her side's win). A persistently negative sum on
    fascist wins would indicate the sign flip is wrong.

Usage:
    python -m analysis.gse_sign_verification <runs-dir> [--n 5] [--out FILE]

`runs-dir` should be one of the runsF2-* folders. The canonical choice is
`runsF2-LLAMA3370B` (the 100-game Llama 3.3 70B run); it can also be
pointed at any other F2 dir as a smoke test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.common import (  # noqa: E402
    ALICE_ID, per_game_record, winner_from_game,
)
from analysis.stateeval_param import (  # noqa: E402
    evaluate, unlocked_powers_for,
)


def load_game(fpath: Path) -> tuple[dict, dict]:
    """Return (raw_game_dict, per_game_record)."""
    with open(fpath) as f:
        g = json.load(f)
    rec = per_game_record(g)
    return g, rec


def pick_fascist_wins(folder: Path, n: int) -> list[Path]:
    """Return paths to N fascist-winning games (Alice fascist/hitler, fascists won)."""
    out = []
    for fpath in sorted(folder.glob("*_summary.json")):
        if "annotat" in fpath.name.lower():
            continue
        try:
            with open(fpath) as f:
                g = json.load(f)
        except Exception:
            continue
        gs = g.get("gameSetting")
        if gs is not None and gs.get("avalonSH") is not None:
            continue
        # Alice role
        players = g.get("players") or []
        if not players or not players[0].get("username", "").startswith("Alice"):
            continue
        alice_role = (players[ALICE_ID].get("role") or "").lower()
        if alice_role not in ("fascist", "hitler"):
            continue
        winner = winner_from_game(g)
        if winner != "fascists":
            continue
        out.append(fpath)
        if len(out) >= n:
            break
    return out


def trace_game(fpath: Path) -> dict:
    g, rec = load_game(fpath)
    if rec is None:
        return dict(error="rec is None")
    players = g.get("players") or []
    alice_role = rec["alice_role"]
    sign = 1.0 if alice_role == "liberal" else -1.0
    winner = winner_from_game(g)

    rows = []
    team_rel_scores = []
    for tr in rec["turn_records"]:
        logged = tr.get("gameStateScore")
        # If the new data has gameStateScore in logs, use it; otherwise recompute
        if logged is None:
            gs_dict = dict(
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
            score_lib = evaluate(gs_dict, tr["true_roles"])
        else:
            score_lib = logged
        score_team = sign * score_lib
        team_rel_scores.append(score_team)
        rows.append(dict(
            r=tr["round_index"],
            pres=tr["president_id"],
            chan=tr["chancellor_id"],
            enacted=tr.get("enacted"),
            lp_before=tr["liberal_policies"],
            fp_before=tr["fascist_policies"],
            score_lib=score_lib,
            score_team=score_team,
            alice_acted=tr["alice_acted"],
        ))

    # Compute deltas + Alice GSIR-total (sum over Alice's action turns only)
    alice_gsir_actions = []
    for i in range(len(rows) - 1):
        delta_team = rows[i + 1]["score_team"] - rows[i]["score_team"]
        rows[i]["delta_team"] = delta_team
        if rows[i]["alice_acted"]:
            alice_gsir_actions.append(delta_team)
    rows[-1]["delta_team"] = None  # last turn has no "next"

    return dict(
        file=fpath.name,
        alice_role=alice_role,
        winner=winner,
        n_rounds=len(rows),
        rows=rows,
        alice_gsir_total=float(np.sum(alice_gsir_actions)) if alice_gsir_actions else 0.0,
        alice_gsir_mean=float(np.mean(alice_gsir_actions)) if alice_gsir_actions else 0.0,
        alice_gsir_n_actions=len(alice_gsir_actions),
        roles={p["username"]: p["role"].lower() for p in players},
    )


def print_trace(t: dict) -> None:
    print(f"\n--- {t['file']}  |  alice={t['alice_role']}  winner={t['winner']}  |  rounds={t['n_rounds']} ---")
    print(f"  Role assignments: {t['roles']}")
    print(f"  {'r':>2}  {'pres':>4}  {'chan':>4}  {'enact':>7}  {'lp/fp':>6}  "
          f"{'score_lib':>10}  {'score_team':>11}  {'Δteam':>10}  {'A':>1}")
    for row in t["rows"]:
        a_mark = "*" if row["alice_acted"] else " "
        d = row.get("delta_team")
        d_str = "    -    " if d is None else f"{d:+9.4f}"
        print(f"  {row['r']:>2}  {row['pres']:>4}  {row['chan']:>4}  "
              f"{(row['enacted'] or ''):>7}  {row['lp_before']}/{row['fp_before']:<3} "
              f"{row['score_lib']:>+10.4f}  {row['score_team']:>+11.4f}  {d_str}  {a_mark}")
    sign_str = "POSITIVE ✓" if t["alice_gsir_total"] > 0 else ("NEGATIVE ✗" if t["alice_gsir_total"] < 0 else "ZERO")
    print(f"  Alice action-turns:     {t['alice_gsir_n_actions']}")
    print(f"  Alice GSIR total:       {t['alice_gsir_total']:+.4f}  ({sign_str})")
    print(f"  Alice GSIR mean/action: {t['alice_gsir_mean']:+.4f}")
    # Expectation: fascist-winning game with alice in (fascist, hitler) => team-positive
    expect_pos = (t["alice_role"] in ("fascist", "hitler") and t["winner"] == "fascists")
    if expect_pos:
        ok = t["alice_gsir_total"] > 0
        print(f"  [sign check] expected POSITIVE for fascist Alice winning fascist game: "
              f"{'PASS' if ok else 'FAIL (potential GSE/GSIR sign bug or Alice played passively)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="runs* folder to inspect")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="optional JSON output path with the full traces")
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"not a directory: {folder}")

    games = pick_fascist_wins(folder, args.n)
    if not games:
        sys.exit(f"no fascist-winning games found in {folder}")

    print(f"Found {len(games)} fascist-winning game(s) in {folder}")
    traces = []
    pass_count = 0
    fail_count = 0
    for fp in games:
        t = trace_game(fp)
        print_trace(t)
        traces.append(t)
        if t["alice_role"] in ("fascist", "hitler") and t["winner"] == "fascists":
            if t["alice_gsir_total"] > 0:
                pass_count += 1
            else:
                fail_count += 1

    print("\n" + "=" * 78)
    print(f"Sign-check summary across {len(traces)} fascist-winning games:")
    print(f"  POSITIVE Alice GSIR (expected): {pass_count}")
    print(f"  NEGATIVE Alice GSIR (unexpected): {fail_count}")
    if fail_count == 0:
        print("  Verdict: sign convention is correct on this sample.")
    else:
        print("  Verdict: at least one fascist-winning game shows a negative")
        print("           Alice GSIR; inspect the trace above for whether Alice")
        print("           was actively responsible or simply absent from key turns.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(folder=folder.name,
                           pass_count=pass_count, fail_count=fail_count,
                           traces=traces), f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
