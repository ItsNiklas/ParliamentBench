"""
Evaluation function mapping a Secret Hitler gamestate to a score in [-1, 1]
+1 = very favourable to Liberals
-1 = very favourable to Fascists

Design summary (see function docstrings for details):
- Input: gamestate dict describing the current board (policies enacted, deck counts, who is president,
  round number, unlocked presidential powers, number of players, true role information for power holders,
  and role-guess information from liberal players).
- Output: float in [-1, 1].
- Internals: modular sub-scores (policies, deck composition, power-holder impact, president suspicion,
  role-consensus/coordination) are computed independently, smoothly normalized using math.tanh functions,
  then aggregated using tunable weights. A global confidence multiplier down-weights outputs when input
  information is sparse.

The file contains:
- evaluate_gamestate(gamestate, config=None) -> float
- helper sub-score functions
- a small "main" block with example gamestates to illustrate usage and expected outputs

Note: This version incorporates advanced balancing techniques and uses true role information for evaluation.

Key improvements stolen from another implementation:
- Dynamic deck weighting that decreases over time (deck matters less in late rounds)
- Truth-aware role accuracy scoring with weighted importance (Hitler > fascist > liberal)
- Hitler election danger scoring (amplified after 3+ fascist policies)
- Power scaling based on number of unlocked powers
- Urgency multipliers for critical game states

Component weights:
- Policy progress (40%): Most critical, with urgency scaling near victory
- Deck composition (dynamic): Starts at 25%, decreases ~1.5% per round
- Powers (20%): Scaled by number of unlocked powers and holder role
- Role accuracy (10%): Truth-aware scoring of liberal identification accuracy
- Hitler danger (5%): Massively amplified (8x) after 3+ fascist policies

Role guess format: {liberal_player: {target_player: "liberal"|"fascist"|"hitler"}}

"""

from __future__ import annotations
from typing import Dict, Any, Sequence
import math


# ----------------------------- GLOBAL CONFIG -----------------------------
# Base weights for each component. Deck weight will be dynamic based on round number.
WEIGHTS = {
    "policy": 0.40,  # Immediate progress toward victory - most critical
    "deck_base": 0.25,  # Base deck weight (will be scaled by round)
    "powers": 0.20,  # Unlocked presidential powers and holder
    "roles": 0.10,  # Role identification accuracy
    "hitler_danger": 0.05,  # Hitler election danger (amplified after 3F policies)
}

# Impact magnitude of specific powers when held by a single player (signed: positive helps Liberals
# if holder is liberal; negative helps Fascists if holder is fascist). These are base magnitudes
# reflecting the true game-changing potential of each power.
POWER_IMPACTS = {
    "execution": 0.85,  # Extremely powerful - can eliminate Hitler or key players
    "investigate": 0.60,  # Very valuable - provides crucial role information
    "policy_peek": 0.35,  # Useful - helps with strategic planning
    "none": 0.0,
}

# Base scaling factor for deck composition impact. Higher values make deck bias more influential.
DECK_INFLUENCE_SCALE = 1.2

# Max policies needed to win for each side (game rule constants).
LIBERAL_WIN_POLICIES = 5
FASCIST_WIN_POLICIES = 6


# ----------------------------- SUB-SCORES -----------------------------


def policy_progress_score(lib_policies: int, fac_policies: int) -> float:
    """
    Compute a policy progress score in [-1,1]. Positive => liberals ahead.

    This score becomes more extreme as either side approaches victory, since being one policy
    away from winning is much more significant than early-game policy differences.
    """
    # Calculate progress ratios
    lp = lib_policies / LIBERAL_WIN_POLICIES
    fp = fac_policies / FASCIST_WIN_POLICIES

    # Basic difference
    base_diff = lp - fp

    # Amplify the score when either side is close to winning
    # Being at 4/5 liberal policies vs 3/6 fascist policies should be heavily positive
    urgency_multiplier = 1.0 + max(lp, fp) * 2.0  # Up to 3x multiplier when someone is close
    amplified_diff = base_diff * urgency_multiplier
    return math.tanh(amplified_diff * 1.2)  # Scale for appropriate sensitivity


def deck_composition_score(deck: Dict[str, int], round_number: int) -> tuple[float, float]:
    """
    Deck bias score in [-1,1] where +1 means deck strongly favors Liberals and -1 strongly favors Fascists.

    Returns both the score and the dynamic weight for this round.
    Deck matters less in later rounds when fewer decisions remain.
    """
    l = deck.get("L", 0)
    f = deck.get("F", 0)
    total = l + f
    if total == 0:
        return 0.0, 0.0

    # Basic bias: proportion difference
    bias = (l - f) / total

    # Dynamic weight: deck matters less as game progresses
    deck_weight = max(0.05, WEIGHTS["deck_base"] - 0.015 * round_number)

    # Size factor: larger remaining decks are more predictive
    size_factor = 0.6 + 0.4 * min(1.0, total / 17.0)

    return math.tanh(bias * size_factor * DECK_INFLUENCE_SCALE), deck_weight


def president_score(unlocked_powers: Sequence[str], president_role: str) -> float:
    """
    Evaluate the powers' net effect from Liberals' perspective based on the true role of the power holder.
    If a power is unlocked and the holder is fascist, it drags the score negative; if the holder is liberal, it helps.
    More unlocked powers amplify the effect.

    Input:
      unlocked_powers: list of power ids, e.g. ["investigate", "execution"]
      president_role: true role of the power holder ("liberal", "fascist", or "hitler")
    Output in [-1,1]
    """
    if not unlocked_powers:
        return 0.0

    # Determine the role factor: +1 for liberal, -1 for fascist/hitler
    if president_role == "liberal":
        role_factor = 1.0
    elif president_role in ["fascist", "hitler"]:
        role_factor = -1.0
    else:
        # Unknown or invalid role -> neutral
        role_factor = 0.0

    # Calculate power impact
    total = 0.3 * role_factor + sum(POWER_IMPACTS.get(p, 0.0) * role_factor for p in unlocked_powers)

    return math.tanh(total)


def role_accuracy_score(role_guesses_by_liberals: Dict[Any, Dict[Any, str]], true_roles: Dict[Any, str]) -> float:
    """
    Evaluate how accurately liberals have identified roles using ground truth.

    Weighted scoring:
    - Correctly identifying Hitler: +1.5 (critical)
    - Correctly identifying fascist: +1.0 (important)
    - Correctly identifying liberal: +0.5 (less critical)
    - Incorrectly identifying Hitler as liberal: -1.0 (very dangerous)
    - Incorrectly identifying fascist as liberal: -1.0 (bad)
    - Incorrectly identifying liberal as fascist: -0.5 (wastes effort)

    Returns score in [-1,1] range.
    """
    if not role_guesses_by_liberals or not true_roles:
        return 0.0

    role_scores = []

    for lib_player, guesses in role_guesses_by_liberals.items():
        for target_player, guessed_role in guesses.items():
            true_role = true_roles.get(target_player)
            if not true_role:
                continue

            if guessed_role == true_role:
                # Correct identification - weighted by importance
                if true_role == "hitler":
                    role_scores.append(1.5)
                elif true_role == "fascist":
                    role_scores.append(1.0)
                else:  # liberal
                    role_scores.append(0.5)
            else:
                # Incorrect identification - penalized by danger
                if true_role == "hitler" and guessed_role == "liberal":
                    role_scores.append(-1.0)  # Very dangerous mistake
                elif true_role == "fascist" and guessed_role == "liberal":
                    role_scores.append(-1.0)  # Bad mistake
                elif true_role == "liberal" and guessed_role in ["fascist", "hitler"]:
                    role_scores.append(-0.5)  # Wasteful suspicion
                else:
                    role_scores.append(-0.3)  # Other misclassifications

    if not role_scores:
        return 0.0

    avg_score = sum(role_scores) / len(role_scores)
    return math.tanh(avg_score)


def hitler_election_danger(fascist_policies: int, role_guesses_by_liberals: Dict[Any, Dict[Any, str]], true_roles: Dict[Any, str]) -> float:
    """
    Evaluate the danger of Hitler being elected chancellor after 3+ fascist policies.

    If fascist policies >= 3 and liberals misidentify Hitler as liberal, this is extremely dangerous.
    Returns negative score (danger to liberals) in this case.
    """
    if fascist_policies < 3 or not role_guesses_by_liberals or not true_roles:
        return 0.0

    # Find Hitler's player ID
    hitler_id = None
    for player_id, role in true_roles.items():
        if role == "hitler":
            hitler_id = player_id
            break

    if not hitler_id:
        return 0.0

    # Check how liberals perceive Hitler
    hitler_perceptions = []
    for lib_player, guesses in role_guesses_by_liberals.items():
        hitler_guess = guesses.get(hitler_id)
        if hitler_guess:
            hitler_perceptions.append(hitler_guess)

    if not hitler_perceptions:
        return 0.0

    # Calculate danger based on liberal perceptions
    liberal_count = sum(1 for guess in hitler_perceptions if guess == "liberal")
    fascist_count = sum(1 for guess in hitler_perceptions if guess in ["fascist", "hitler"])

    # If majority think Hitler is liberal = very dangerous
    if liberal_count > fascist_count:
        danger_score = -1.0
    elif fascist_count > liberal_count:
        danger_score = 0.5  # Good - correctly suspicious
    else:
        danger_score = -0.3  # Uncertain = some danger

    # Scale by how late in the game (more fascist policies = more danger)
    urgency_multiplier = min(2.0, fascist_policies / 3.0)

    return math.tanh(danger_score * urgency_multiplier)


# ----------------------------- MAIN AGGREGATOR -----------------------------


def evaluate_gamestate(gamestate: Dict[str, Any], true_roles: Dict[str, str] = None, debug: bool = False) -> float:
    """
    Primary evaluation entry point with improved balancing from another implementation.

    Expected gamestate keys (examples):
      - 'liberal_policies': int
      - 'fascist_policies': int
      - 'deck': {'L': int, 'F': int}
      - 'president': player id (hashable)
      - 'round': int (1-based)
      - 'unlocked_powers': list of power ids (strings)
      - 'president_role': str ("liberal", "fascist", or "hitler")
      - 'num_players': int
      - 'role_guesses_by_liberals': {lib_player: {target_player: "liberal"|"fascist"|"hitler"}}

    true_roles: Dict mapping player_id -> role for ground truth evaluation

    Returns float in [-1,1] (positive -> Liberal advantage)
    """
    lp = int(gamestate.get("liberal_policies", 0))
    fp = int(gamestate.get("fascist_policies", 0))
    deck = gamestate.get("deck", {"L": 0, "F": 0})
    rnd = int(gamestate.get("round", 1))
    unlocked_powers = gamestate.get("unlocked_powers", [])
    president_role = gamestate.get("president_role", "liberal")  # Default assumption
    role_guesses = gamestate.get("role_guesses_by_liberals", {})

    if debug:
        print(f"  DEBUG: Policies {lp}L-{fp}F, Deck {deck['L']}L-{deck['F']}F, Round {rnd}, Powers {unlocked_powers}")

    # Subscores with improved balancing
    s_policy = policy_progress_score(lp, fp)  # [-1,1]
    s_deck, deck_weight = deck_composition_score(deck, rnd)  # Dynamic weight based on round
    s_powers = president_score(unlocked_powers, president_role)  # [-1,1]

    # Truth-aware role evaluation (fallback to 0 if no true_roles provided)
    if true_roles:
        s_roles = role_accuracy_score(role_guesses, true_roles)  # [-1,1]
        s_hitler = hitler_election_danger(fp, role_guesses, true_roles)  # [-1,1]
    else:
        s_roles = 0.0
        s_hitler = 0.0

    if debug:
        print(f"  DEBUG: Scores - Policy:{s_policy:+.3f} Deck:{s_deck:+.3f}(w:{deck_weight:.3f}) Powers:{s_powers:+.3f} Roles:{s_roles:+.3f} Hitler:{s_hitler:+.3f}")

    # Dynamic weighted aggregation with weight redistribution
    hitler_weight = WEIGHTS["hitler_danger"]
    if fp >= 3:
        hitler_weight *= 4.0  # Massive amplification after 3F policies

    # Collect active components (non-zero scores) and their weights
    components = [
        ("policy", WEIGHTS["policy"], s_policy),
        ("deck", deck_weight, s_deck),
        ("powers", WEIGHTS["powers"], s_powers),
        ("roles", WEIGHTS["roles"], s_roles),
        ("hitler", hitler_weight, s_hitler),
    ]
    
    # Separate active and inactive components
    active_components = [(name, weight, score) for name, weight, score in components if abs(score) > 1e-6]
    inactive_weight = sum(weight for name, weight, score in components if abs(score) <= 1e-6)
    
    # Redistribute inactive weight proportionally to active components
    if active_components and inactive_weight > 0:
        active_weight_sum = sum(weight for _, weight, _ in active_components)
        if active_weight_sum > 0:
            redistribution_factor = (active_weight_sum + inactive_weight) / active_weight_sum
            # Apply redistributed weights
            raw_score = sum(weight * redistribution_factor * score for _, weight, score in active_components)
        else:
            raw_score = 0.0
    else:
        # Standard calculation when all components are active or no redistribution needed
        raw_score = sum(weight * score for _, weight, score in components)

    # Apply confidence multiplier
    confidence = (math.tanh(rnd / 5) + 1.2) / 2.0  # Map from [-1,1] to [0,1]
    final = raw_score * confidence

    if debug:
        if inactive_weight > 0:
            active_names = [name for name, _, _ in active_components]
            print(f"  DEBUG: Weight redistribution: {inactive_weight:.3f} redistributed to {active_names}")
        print(f"  DEBUG: Raw:{raw_score:+.3f} Confidence:{confidence:.3f} Final:{final:+.3f} -> {math.tanh(final):+.3f}")

    return math.tanh(final)


# ----------------------------- EXAMPLES / QUICK TESTS -----------------------------
if __name__ == "__main__":
    examples = []
    true_roles_examples = []

    # Example 1: very early game, neutral
    examples.append(
        {
            "liberal_policies": 0,
            "fascist_policies": 0,
            "deck": {"L": 6, "F": 11},  # 6-0=6L, 11-0=11F
            "president": "P0",
            "round": 1,
            "unlocked_powers": [],
            "president_role": "liberal",
            "num_players": 5,
            "role_guesses_by_liberals": {},
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "fascist", "P3": "liberal", "P4": "hitler"})

    # Example 2: fascist momentum, execution unlocked and holder is fascist
    examples.append(
        {
            "liberal_policies": 1,
            "fascist_policies": 3,
            "deck": {"L": 5, "F": 8},  # 6-1=5L, 11-3=8F
            "president": "P3",
            "round": 7,
            "unlocked_powers": ["execution"],
            "president_role": "fascist",
            "num_players": 7,
            "role_guesses_by_liberals": {
                "L1": {"P3": "fascist", "P2": "liberal"},
                "L2": {"P3": "fascist", "P1": "liberal"},
            },
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "liberal", "P3": "fascist", "P4": "fascist", "P5": "hitler", "P6": "liberal"})

    # Example 3: liberals have strong consensus who the fascist is
    examples.append(
        {
            "liberal_policies": 2,
            "fascist_policies": 2,
            "deck": {"L": 4, "F": 9},  # 6-2=4L, 11-2=9F
            "president": "P4",
            "round": 6,
            "unlocked_powers": [],
            "president_role": "liberal",
            "num_players": 5,
            "role_guesses_by_liberals": {
                "L1": {"P2": "fascist", "P3": "liberal", "P4": "liberal"},
                "L2": {"P2": "fascist", "P3": "liberal", "P1": "liberal"},
                "L3": {"P2": "fascist", "P4": "liberal", "P1": "liberal"},
            },
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "fascist", "P3": "liberal", "P4": "hitler"})

    # Example 4: Critical late-game scenario - liberals one policy away from winning with good deck
    examples.append(
        {
            "liberal_policies": 4,
            "fascist_policies": 2,
            "deck": {"L": 2, "F": 9},  # 6-4=2L, 11-2=9F
            "president": "P1",
            "round": 10,
            "unlocked_powers": [],
            "president_role": "liberal",
            "num_players": 5,
            "role_guesses_by_liberals": {
                "L1": {"P2": "fascist", "P3": "hitler"},
                "L2": {"P2": "fascist", "P3": "hitler"},
            },
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "fascist", "P3": "hitler", "P4": "liberal"})

    # Example 5: Dire fascist scenario - fascists close to winning with terrible deck for liberals
    examples.append(
        {
            "liberal_policies": 1,
            "fascist_policies": 5,
            "deck": {"L": 5, "F": 6},  # 6-1=5L, 11-5=6F
            "president": "P3",
            "round": 12,
            "unlocked_powers": ["execution"],
            "president_role": "fascist",
            "num_players": 7,
            "role_guesses_by_liberals": {
                "L1": {"P3": "fascist"},
            },
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "liberal", "P3": "fascist", "P4": "fascist", "P5": "hitler", "P6": "liberal"})

    # Example 6: Hitler danger scenario - 3F policies and liberals think Hitler is liberal
    examples.append(
        {
            "liberal_policies": 1,
            "fascist_policies": 3,
            "deck": {"L": 5, "F": 8},  # 6-1=5L, 11-3=8F
            "president": "P2",
            "round": 8,
            "unlocked_powers": ["investigate"],
            "president_role": "liberal",
            "num_players": 5,
            "role_guesses_by_liberals": {
                "L1": {"P3": "liberal", "P4": "fascist"},  # WRONG - thinks Hitler is liberal!
                "L2": {"P3": "liberal", "P4": "fascist"},  # WRONG - thinks Hitler is liberal!
            },
        }
    )
    true_roles_examples.append({"P0": "liberal", "P1": "liberal", "P2": "liberal", "P3": "hitler", "P4": "fascist"})

    # Expected scores and descriptions for each example
    descriptions = [
        "Early game (R1): 0L-0F policies, starting deck (6L-11F), no powers, no role info. Should be slightly fascist-favored due to deck composition.",
        "Mid-game crisis (R7): 1L-3F policies, bad deck (5L-8F), fascist has execution power, liberals correctly identify fascist president. Should be moderately fascist-favored.",
        "Mid-game (R6): 2L-2F policies, fascist-heavy deck (4L-9F), liberal president, liberals correctly identify fascist. Should be moderately fascist-favored due to deck.",
        "Late game advantage (R10): 4L-2F policies (libs 1 away!), but terrible deck (2L-9F), liberals correctly identify Hitler and fascist. Should be slightly liberal-favored due to policy progress.",
        "Dire situation (R12): 1L-5F policies (fascists 1 away!), slightly fascist deck (5L-6F), fascist has execution. Should be strongly fascist-favored.",
        "Hitler danger (R8): 1L-3F policies, bad deck (5L-8F), but liberals misidentify Hitler as liberal after 3F! Should be strongly fascist-favored.",
    ]

    expected_scores = [-0.2, -0.3, -0.2, 0.4, -0.8, -0.6]  # Expected scores in 0.1 increments

    # Set to True to see detailed scoring breakdown
    SHOW_DEBUG = True

    print("=== Secret Hitler Game State Evaluation ===\n")

    for i, (ex, roles, desc, expected) in enumerate(zip(examples, true_roles_examples, descriptions, expected_scores), 1):
        print(f"Example {i}")
        print(examples[i-1])
        score = evaluate_gamestate(ex, roles, debug=SHOW_DEBUG)
        diff = abs(score - expected)
        accuracy = "✓" if diff <= 0.1 else "✗" if diff <= 0.2 else "✗✗"
        print(f"  State: {desc}")
        # print(f"  Expected: ~{expected:+.1f} {accuracy}")
        print(f"  Actual:   {score:+.3f}")
        # print(f"  Actual:   {score:+.3f} (diff: {diff:.3f})")
        print()

    # Summary
    total_diff = sum(abs(evaluate_gamestate(ex, roles) - expected) for ex, roles, expected in zip(examples, true_roles_examples, expected_scores))
    avg_diff = total_diff / len(examples)
    print(f"Average prediction error: {avg_diff:.3f}")
    print("✓ = within 0.1, ✗ = within 0.2, ✗✗ = over 0.2 difference")