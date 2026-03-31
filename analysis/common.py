"""Shared loaders for the bootstrap-CI, DRR-decomposition, and
GSIR-perturbation analyses.

Reads the per-game JSON dumps written by `parliamentbench.game_engine` and
extracts the per-game records each analysis needs:
  - alice_role / alice_won
  - per-round opponent perception of Alice  (for the DRR decomposition)
  - per-round gameStateScore  (for the GSIR perturbation recompute)
  - per-round "alice-acted" indicator and team-relative score delta  (GSIR base)

Run directories are looked up under RUNS_ROOT (default: the repository root).
Set the RUNS_ROOT environment variable to point at wherever the recorded runs
live; they are not shipped with this repository.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from analysis.model_registry import MODEL_REGISTRY, is_baseline

# Repository root — where this package and analysis/results live.
ROOT = Path(__file__).resolve().parent.parent

# Root directory holding the recorded game-run folders. Override with RUNS_ROOT.
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", ROOT))

# Where the analyses write their JSON output.
RESULTS_DIR = ROOT / "analysis" / "results"

ALICE_ID = 0


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def model_folders(include_baselines: bool = True) -> list[Path]:
    """All runsF2-* folders registered in MODEL_REGISTRY and present on disk."""
    out = []
    for key, info in MODEL_REGISTRY.items():
        if not key.startswith("runsF2"):
            continue
        if info.get("abliterated", False):
            continue
        if not include_baselines and is_baseline(key):
            continue
        p = RUNS_ROOT / key
        if p.is_dir():
            out.append(p)
    return out


def display_name(folder: Path) -> str:
    info = MODEL_REGISTRY.get(folder.name)
    return info["name"] if info else folder.name


# ---------------------------------------------------------------------------
# Game parsing
# ---------------------------------------------------------------------------

def alice_record(players: list[dict]) -> tuple[str | None, str | None, int | None]:
    """Return (alice_role_lower, alice_username, alice_pid)."""
    for i, p in enumerate(players):
        if p.get("username", "").startswith("Alice"):
            return p.get("role", "").lower(), p.get("username"), i
    return None, None, None


def winner_from_game(g: dict) -> str | None:
    """Return 'liberals'|'fascists' from gameEndReason or policy counts."""
    counts = g.get("policyCountsAtEnd") or {}
    lp = counts.get("liberal", 0)
    fp = counts.get("fascist", 0)
    end_reason = (g.get("gameEndReason") or "").lower()
    if "hitler" in end_reason and "chancellor" in end_reason:
        return "fascists"
    if "hitler" in end_reason and ("killed" in end_reason or "shot" in end_reason or "execut" in end_reason):
        return "liberals"
    if lp >= 5:
        return "liberals"
    if fp >= 6:
        return "fascists"
    # fallback: more policies wins (incomplete games)
    if lp > fp:
        return "liberals"
    if fp > lp:
        return "fascists"
    return None


# Same parsing strategy used by eval/deception_analysis.py: keep "unknown".
ROLE_TOKEN_RE = re.compile(r"\b(liberal|fascist|hitler|unknown)", re.IGNORECASE)

def parse_perception_of(text: str, target_username: str) -> str:
    """Return lower-case perceived role of *target_username* in a rapidAssessment text.

    Strategy mirrors deception_analysis._parse_rapid_assessment: locate the
    first occurrence of the target name (whole-word, case-insensitive) and scan
    forward for the first role keyword. Returns 'unknown' if the name is
    mentioned but no role keyword follows, and 'unmentioned' if not mentioned.
    """
    if not text or not target_username:
        return "unmentioned"
    m = re.search(r"\b" + re.escape(target_username) + r"\b", text, re.IGNORECASE)
    if m is None:
        return "unmentioned"
    after = text[m.start():]
    rm = ROLE_TOKEN_RE.search(after)
    if rm is None:
        return "unknown"
    return rm.group(1).lower()


# ---------------------------------------------------------------------------
# Per-game extraction
# ---------------------------------------------------------------------------

def per_game_record(g: dict) -> dict | None:
    """Extract a compact per-game record. Returns None for unusable games."""
    players = g.get("players") or []
    if not players:
        return None
    alice_role, alice_name, alice_pid = alice_record(players)
    if alice_role is None:
        return None
    winner = winner_from_game(g)
    if winner is None:
        return None
    if alice_role == "liberal":
        alice_won = winner == "liberals"
    else:
        alice_won = winner == "fascists"

    logs = g.get("logs") or []
    # Require enough rounds: skip ultra-short games
    if len(logs) < 4:
        return None

    # Role lookup by player index
    role_by_pid = {i: (p.get("role") or "").lower() for i, p in enumerate(players)}
    name_by_pid = {i: p.get("username") for i, p in enumerate(players)}

    # ---- DRR per-round perceptions of Alice by *liberal* opponents ----
    drr_perceptions: list[str] = []  # one entry per (liberal opponent, round) when alice is fas/hit
    if alice_role in ("fascist", "hitler"):
        for log in logs:
            ra = log.get("rapidAssessments") or {}
            for pid_str, txt in ra.items():
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    continue
                if pid == alice_pid:
                    continue
                if role_by_pid.get(pid) != "liberal":
                    continue
                perc = parse_perception_of(txt, alice_name)
                if perc == "unmentioned":
                    continue
                drr_perceptions.append(perc)

    # ---- Per-turn raw fields for GSIR recompute (perturbation analysis) ----
    # We store enough to recompute gameStateScore via stateeval with perturbed constants.
    turn_records: list[dict] = []
    lib_count = 0
    fas_count = 0
    n_players = len(players)
    # Liberal player ids needed to gate role_guesses_by_liberals.
    liberal_pids = {pid for pid, r in role_by_pid.items() if r == "liberal"}

    for idx, log in enumerate(logs):
        # gamestate snapshot represents the state at the start of this turn,
        # before this turn's policy is enacted — match the simulator's behavior
        # where calculate_gamestate_score is called when the chancellor is voted.
        # But the simulator actually logs scores at the *end* of the turn (after policy enact)
        # judging by HitlerGame.py:187. Track both ways: we will use enactedPolicy of
        # previous turn to update counts before scoring this turn.
        # Build deck composition from the recorded deckState (post-enact for this log).
        deck = log.get("deckState") or []
        deck_l = sum(1 for c in deck if c == "liberal")
        deck_f = sum(1 for c in deck if c == "fascist")

        # Parse role guesses by liberals from this log's rapidAssessments.
        # Map by target *username* per player. Used by stateeval as
        # {lib_pid: {target_pid: role}}.
        ra = log.get("rapidAssessments") or {}
        role_guesses: dict[int, dict[int, str]] = {}
        if ra:
            other_names = [n for pid, n in name_by_pid.items()]
            for pid_str, txt in ra.items():
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    continue
                if pid not in liberal_pids:
                    continue
                # parse perceptions for every opponent
                guesses: dict[int, str] = {}
                for tpid, tname in name_by_pid.items():
                    if tpid == pid:
                        continue
                    perc = parse_perception_of(txt, tname)
                    if perc in ("liberal", "fascist", "hitler"):
                        guesses[tpid] = perc
                if guesses:
                    role_guesses[pid] = guesses

        # Update running policy counts using the *enactedPolicy* of THIS turn,
        # so we can record the state both *before* and *after* this turn.
        enacted = log.get("enactedPolicy")
        pres_id = log.get("presidentId")
        chan_id = log.get("chancellorId")

        # State BEFORE this turn's policy enacts (matches the deck before draw).
        # But the recorded deckState in each log is *after* the policy was drawn out
        # (since the cards are drawn during the turn). We treat the log as the snapshot
        # used by the simulator's calculate_gamestate_score call.
        # For the perturbation analysis, exact alignment with the original log score
        # is not necessary; what matters is internal consistency across constants.
        # Previous-turn role_guesses are what the simulator passed into the
        # state evaluator at turn start (see HitlerGame.py:187 +
        # HitlerGameState.calculate_gamestate_score signature).
        prev_role_guesses = turn_records[-1]["role_guesses"] if turn_records else {}
        turn_records.append(dict(
            round_index=idx + 1,
            liberal_policies=lib_count,  # before enact
            fascist_policies=fas_count,
            deck_l=deck_l,
            deck_f=deck_f,
            president_id=pres_id,
            chancellor_id=chan_id,
            president_role=role_by_pid.get(pres_id) or "liberal",
            n_players=n_players,
            role_guesses=role_guesses,          # this turn's parsed RA
            role_guesses_prev=prev_role_guesses,  # used by stateeval at this turn
            true_roles=role_by_pid,
            gameStateScore=log.get("gameStateScore"),
            enacted=enacted,
            alice_acted=(pres_id == ALICE_ID or chan_id == ALICE_ID),
        ))

        if enacted == "liberal":
            lib_count += 1
        elif enacted == "fascist":
            fas_count += 1

    return dict(
        alice_role=alice_role,
        alice_won=alice_won,
        drr_perceptions=drr_perceptions,
        turn_records=turn_records,
        n_logs=len(logs),
    )


def load_records(folder: Path) -> list[dict]:
    out = []
    for fpath in sorted(folder.glob("*_summary.json")):
        if "annotat" in fpath.name.lower():
            continue
        try:
            with open(fpath) as f:
                g = json.load(f)
        except Exception:
            continue
        # Skip Avalon
        gs = g.get("gameSetting")
        if gs is not None and gs.get("avalonSH") is not None:
            continue
        rec = per_game_record(g)
        if rec is not None:
            out.append(rec)
    return out


def model_records(include_baselines: bool = True) -> dict[str, list[dict]]:
    """Return {display_name: [game records]} for every registered model on disk."""
    out: dict[str, list[dict]] = {}
    for fld in model_folders(include_baselines=include_baselines):
        recs = load_records(fld)
        if recs:
            out[display_name(fld)] = recs
    return out
