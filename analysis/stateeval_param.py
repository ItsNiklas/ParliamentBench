"""Parameterised version of parliamentbench.metric.stateeval.evaluate_gamestate.

Mirrors the math in parliamentbench/metric/stateeval.py but exposes every hand-tuned
constant as a key in a `params` dict so we can perturb them in the
GSIR-perturbation sensitivity analysis.

The defaults are exactly the values in stateeval.py. Calling
`evaluate(gamestate, true_roles)` with no params returns the same number the
simulator would have computed at run-time.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_PARAMS: dict[str, float] = {
    # policy_progress_score
    "policy_tanh_scale": 1.2,
    "policy_urgency_mult": 2.0,
    # deck_composition_score
    "deck_size_floor": 0.6,
    "deck_size_slope": 0.4,
    "deck_size_anchor": 17.0,
    "deck_influence_scale": 1.2,
    "deck_weight_decay": 0.015,
    "deck_weight_floor": 0.05,
    # president_score
    "pres_role_factor": 0.3,
    "power_execution": 0.85,
    "power_investigate": 0.60,
    "power_peek": 0.35,
    # role_accuracy_score
    "role_correct_hitler": 1.5,
    "role_correct_fascist": 1.0,
    "role_correct_liberal": 0.5,
    "role_wrong_dangerous": -1.0,
    "role_wrong_libsuspect": -0.5,
    "role_wrong_other": -0.3,
    # hitler_election_danger
    "hd_lib_consensus_score": -1.0,
    "hd_fas_consensus_score": 0.5,
    "hd_uncertain_score": -0.3,
    # base weights
    "weight_policy": 0.40,
    "weight_deck_base": 0.25,
    "weight_powers": 0.20,
    "weight_roles": 0.10,
    "weight_hitler": 0.05,
    "hitler_weight_3f_mult": 4.0,
    # confidence factor (math.tanh(rnd / d) + a) / b
    "conf_round_denom": 5.0,
    "conf_offset": 1.2,
    "conf_divisor": 2.0,
    # constants in HitlerFactory referenced inside policy_progress_score
    "lib_win_policies": 5.0,
    "fas_win_policies": 6.0,
}


# The 12 constants targeted by the perturbation sensitivity analysis. The
# remaining DEFAULT_PARAMS are exposed for completeness but only this set is
# perturbed.
SPEC_PARAM_KEYS = [
    "policy_tanh_scale",     # 1.2
    "deck_size_floor",       # 0.6
    "deck_size_slope",       # 0.4
    "conf_round_denom",      # 5
    "power_execution",       # 0.85
    "power_investigate",     # 0.60
    "power_peek",            # 0.35
    "role_correct_liberal",  # 0.5
    "role_wrong_other",      # -0.3
    "role_wrong_dangerous",  # -1.0
    "conf_offset",           # 1.2 — "0.6 in confidence factor" (1.2/2.0 == 0.6 floor)
    "conf_divisor",          # 2.0 — "0.5 in confidence factor" (1/2)
]


def policy_progress_score(lp: int, fp: int, p: dict) -> float:
    lp_ratio = lp / p["lib_win_policies"]
    fp_ratio = fp / p["fas_win_policies"]
    base_diff = lp_ratio - fp_ratio
    urgency = 1.0 + max(lp_ratio, fp_ratio) * p["policy_urgency_mult"]
    return math.tanh(base_diff * urgency * p["policy_tanh_scale"])


def deck_composition_score(deck: dict, rnd: int, p: dict) -> tuple[float, float]:
    l = deck.get("L", 0)
    f = deck.get("F", 0)
    total = l + f
    if total == 0:
        return 0.0, 0.0
    bias = (l - f) / total
    deck_weight = max(p["deck_weight_floor"], p["weight_deck_base"] - p["deck_weight_decay"] * rnd)
    size_factor = p["deck_size_floor"] + p["deck_size_slope"] * min(1.0, total / p["deck_size_anchor"])
    return math.tanh(bias * size_factor * p["deck_influence_scale"]), deck_weight


def president_score(unlocked_powers, president_role: str, p: dict) -> float:
    if not unlocked_powers:
        return 0.0
    if president_role == "liberal":
        role_factor = 1.0
    elif president_role in ("fascist", "hitler"):
        role_factor = -1.0
    else:
        role_factor = 0.0
    impacts = {
        "execution": p["power_execution"],
        "investigate": p["power_investigate"],
        "policy_peek": p["power_peek"],
        "none": 0.0,
    }
    total = p["pres_role_factor"] * role_factor + sum(impacts.get(pw, 0.0) * role_factor for pw in unlocked_powers)
    return math.tanh(total)


def role_accuracy_score(role_guesses_by_liberals, true_roles, p: dict) -> float:
    if not role_guesses_by_liberals or not true_roles:
        return 0.0
    scores = []
    for lib_player, guesses in role_guesses_by_liberals.items():
        for target, guess in guesses.items():
            true_role = true_roles.get(target)
            if not true_role:
                continue
            if guess == true_role:
                if true_role == "hitler":
                    scores.append(p["role_correct_hitler"])
                elif true_role == "fascist":
                    scores.append(p["role_correct_fascist"])
                else:
                    scores.append(p["role_correct_liberal"])
            else:
                if true_role == "hitler" and guess == "liberal":
                    scores.append(p["role_wrong_dangerous"])
                elif true_role == "fascist" and guess == "liberal":
                    scores.append(p["role_wrong_dangerous"])
                elif true_role == "liberal" and guess in ("fascist", "hitler"):
                    scores.append(p["role_wrong_libsuspect"])
                else:
                    scores.append(p["role_wrong_other"])
    if not scores:
        return 0.0
    return math.tanh(sum(scores) / len(scores))


def hitler_election_danger(fp: int, role_guesses_by_liberals, true_roles, p: dict) -> float:
    if fp < 3 or not role_guesses_by_liberals or not true_roles:
        return 0.0
    hitler_id = None
    for pid, r in true_roles.items():
        if r == "hitler":
            hitler_id = pid
            break
    if hitler_id is None:
        return 0.0
    hitler_perc = []
    for lib_player, guesses in role_guesses_by_liberals.items():
        g = guesses.get(hitler_id)
        if g:
            hitler_perc.append(g)
    if not hitler_perc:
        return 0.0
    lib_c = sum(1 for g in hitler_perc if g == "liberal")
    fas_c = sum(1 for g in hitler_perc if g in ("fascist", "hitler"))
    if lib_c > fas_c:
        s = p["hd_lib_consensus_score"]
    elif fas_c > lib_c:
        s = p["hd_fas_consensus_score"]
    else:
        s = p["hd_uncertain_score"]
    urgency = min(2.0, fp / 3.0)
    return math.tanh(s * urgency)


def evaluate(gamestate: dict[str, Any], true_roles: dict[Any, str], params: dict | None = None) -> float:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    lp = int(gamestate.get("liberal_policies", 0))
    fp = int(gamestate.get("fascist_policies", 0))
    deck = gamestate.get("deck", {"L": 0, "F": 0})
    rnd = int(gamestate.get("round", 1))
    unlocked = gamestate.get("unlocked_powers", [])
    pres_role = gamestate.get("president_role", "liberal")
    role_guesses = gamestate.get("role_guesses_by_liberals", {})

    s_policy = policy_progress_score(lp, fp, p)
    s_deck, deck_weight = deck_composition_score(deck, rnd, p)
    s_powers = president_score(unlocked, pres_role, p)
    s_roles = role_accuracy_score(role_guesses, true_roles, p) if true_roles else 0.0
    s_hitler = hitler_election_danger(fp, role_guesses, true_roles, p) if true_roles else 0.0

    hitler_weight = p["weight_hitler"]
    if fp >= 3:
        hitler_weight *= p["hitler_weight_3f_mult"]
    components = [
        ("policy", p["weight_policy"], s_policy),
        ("deck", deck_weight, s_deck),
        ("powers", p["weight_powers"], s_powers),
        ("roles", p["weight_roles"], s_roles),
        ("hitler", hitler_weight, s_hitler),
    ]
    active = [(n, w, s) for n, w, s in components if abs(s) > 1e-6]
    inactive_w = sum(w for n, w, s in components if abs(s) <= 1e-6)
    if active and inactive_w > 0:
        active_w_sum = sum(w for _, w, _ in active)
        if active_w_sum > 0:
            redist = (active_w_sum + inactive_w) / active_w_sum
            raw = sum(w * redist * s for _, w, s in active)
        else:
            raw = 0.0
    else:
        raw = sum(w * s for _, w, s in components)
    conf = (math.tanh(rnd / p["conf_round_denom"]) + p["conf_offset"]) / p["conf_divisor"]
    return math.tanh(raw * conf)


# Power lookup mirrors HitlerGameState._get_unlocked_powers but for 5 players
# (sim.sh always runs 5-player games). 5 players uses
# fascist_track_actions = [None, None, "policy", "kill", "kill", None]
ACTION_TABLE_5P = [None, None, "policy", "kill", "kill", None]

ACTION_TO_POWER = {"kill": "execution", "inspect": "investigate", "policy": "policy_peek", "choose": "choose_president"}


def unlocked_powers_for(fp: int) -> list[str]:
    """Return the unlocked powers list given current fascist_policies (5-player game)."""
    if fp <= 0 or fp > len(ACTION_TABLE_5P):
        return []
    out = []
    for i in range(fp):
        a = ACTION_TABLE_5P[i] if i < len(ACTION_TABLE_5P) else None
        if a is None:
            continue
        mapped = ACTION_TO_POWER.get(a)
        if mapped:
            out.append(mapped)
    return out
