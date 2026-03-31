"""Runtime benchmark metrics for ParliamentBench.

This module computes paper-aligned benchmark metrics from generated game summaries:
- Win Rate: overall and by Alice role.
- Game Outcomes: end-condition breakdown and average rounds.
- GSIR (Game-State Impact Rate): cumulative game-state impact by Alice's role.
- RIA (Role Identification Accuracy): role assessment accuracy.
- DRR (Deception Retention Rate): deception success in fascist/hitler games.
- Voting: approval rates (overall/phase), defensive vote accuracy.
- Endorsement/Agreeableness: received yes-votes when president and vote tendencies.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


ALICE_PREFIX = "Alice"
ROLES = ("liberal", "fascist", "hitler")
PHASES = ("early", "mid", "late")


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _find_alice(game: dict[str, Any]) -> tuple[int | None, str | None]:
    players = game.get("players", [])
    for idx, player in enumerate(players):
        username = player.get("username", "")
        if isinstance(username, str) and username.startswith(ALICE_PREFIX):
            return idx, player.get("role", "").lower()
    return None, None


def _normalize_reason(reason: str | None) -> str:
    normalized = (reason or "").strip().lower()
    if normalized in {
        "liberal_policy",
        "hitler_killed",
        "fascist_policy",
        "hitler_chancellor",
        "no_clear_winner",
    }:
        return normalized
    return "other"


def _winner_from_reason(reason: str) -> str:
    if reason in {"liberal_policy", "hitler_killed"}:
        return "liberals"
    if reason in {"fascist_policy", "hitler_chancellor"}:
        return "fascists"
    return "unknown"


def _belief_score(belief: str, true_role: str) -> float:
    if belief == true_role:
        return 1.0
    if {belief, true_role} == {"fascist", "hitler"}:
        return 0.5
    return 0.0


def _parse_rapid_assessment(
    text: str,
    player_names: list[str] | None = None,
    *,
    keep_unknown: bool = False,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if not text or not player_names:
        return result

    for name in player_names:
        matches = list(re.finditer(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE))
        if not matches:
            continue
        search_start = matches[0].start()
        match = re.search(
            r"\b(liberal|fascist|hitler|unknown)\b",
            text[search_start:],
            re.IGNORECASE,
        )
        if match is None:
            continue
        role = match.group(1).lower()
        if role == "unknown" and not keep_unknown:
            continue
        result[name] = role
    return result


def _phase_for_round(round_idx: int) -> str:
    if round_idx <= 3:
        return "early"
    if round_idx <= 7:
        return "mid"
    return "late"


def _empty_vote_bucket() -> dict[str, Any]:
    return {
        "ja": 0,
        "total": 0,
        "danger_total": 0,
        "danger_nein": 0,
        "danger_ja": 0,
        "phases": {
            "early": {"ja": 0, "total": 0},
            "mid": {"ja": 0, "total": 0},
            "late": {"ja": 0, "total": 0},
        },
    }


def _record_vote(bucket: dict[str, Any], voted_ja: bool, phase: str) -> None:
    bucket["total"] += 1
    if voted_ja:
        bucket["ja"] += 1

    phase_data = bucket["phases"][phase]
    phase_data["total"] += 1
    if voted_ja:
        phase_data["ja"] += 1


def _record_danger_vote(bucket: dict[str, Any], voted_ja: bool) -> None:
    bucket["danger_total"] += 1
    if voted_ja:
        bucket["danger_ja"] += 1
    else:
        bucket["danger_nein"] += 1


def _vote_bucket_report(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": _safe_pct(bucket["ja"], bucket["total"]),
        "by_phase": {
            phase: _safe_pct(bucket["phases"][phase]["ja"], bucket["phases"][phase]["total"])
            for phase in PHASES
        },
        "counts": {
            "ja": bucket["ja"],
            "total": bucket["total"],
            "by_phase": {
                phase: {
                    "ja": bucket["phases"][phase]["ja"],
                    "total": bucket["phases"][phase]["total"],
                }
                for phase in PHASES
            },
        },
    }


def _danger_bucket_report(bucket: dict[str, Any], key: str) -> dict[str, Any]:
    numerator = bucket["danger_nein"] if key == "vote_accuracy" else bucket["danger_ja"]
    return {
        "overall": _safe_pct(numerator, bucket["danger_total"]),
        "counts": {
            "danger_total": bucket["danger_total"],
            "danger_nein": bucket["danger_nein"],
            "danger_ja": bucket["danger_ja"],
        },
    }


def compute_win_rates(games: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"total": 0, "wins": 0}
    role_counts: dict[str, dict[str, int]] = {
        "liberal": {"total": 0, "wins": 0},
        "fascist": {"total": 0, "wins": 0},
        "hitler": {"total": 0, "wins": 0},
    }

    for game in games:
        _, alice_role = _find_alice(game)
        if alice_role not in role_counts:
            continue

        reason = _normalize_reason(game.get("gameEndReason"))
        winner = _winner_from_reason(reason)
        if winner == "unknown":
            continue

        counts["total"] += 1
        role_counts[alice_role]["total"] += 1

        alice_won = (alice_role == "liberal" and winner == "liberals") or (
            alice_role in {"fascist", "hitler"} and winner == "fascists"
        )
        if alice_won:
            counts["wins"] += 1
            role_counts[alice_role]["wins"] += 1

    return {
        "win_rate": {
            "overall": _safe_pct(counts["wins"], counts["total"]),
            "by_role": {
                role: _safe_pct(stats["wins"], stats["total"])
                for role, stats in role_counts.items()
            },
            "counts": {
                "wins": counts["wins"],
                "total": counts["total"],
                "by_role": role_counts,
            },
        }
    }


def compute_game_outcomes(games: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    reason_counts_by_role: dict[str, dict[str, int]] = {
        role: defaultdict(int) for role in ROLES
    }
    rounds_all: list[int] = []
    rounds_by_role: dict[str, list[int]] = {role: [] for role in ROLES}

    for game in games:
        _, alice_role = _find_alice(game)
        if alice_role not in ROLES:
            continue

        reason = _normalize_reason(game.get("gameEndReason"))
        reason_counts[reason] += 1
        reason_counts_by_role[alice_role][reason] += 1

        logs = game.get("logs", [])
        rounds = len(logs) if isinstance(logs, list) else 0
        rounds_all.append(rounds)
        rounds_by_role[alice_role].append(rounds)

    total = sum(reason_counts.values())

    def _avg(values: list[int]) -> float:
        if not values:
            return 0.0
        return float(sum(values)) / len(values)

    all_reasons = [
        "liberal_policy",
        "hitler_killed",
        "fascist_policy",
        "hitler_chancellor",
        "no_clear_winner",
        "other",
    ]

    return {
        "game_outcomes": {
            "overall": {
                reason: _safe_pct(reason_counts[reason], total) for reason in all_reasons
            },
            "by_role": {
                role: {
                    reason: _safe_pct(
                        reason_counts_by_role[role][reason],
                        sum(reason_counts_by_role[role].values()),
                    )
                    for reason in all_reasons
                }
                for role in ROLES
            },
            "counts": {
                "overall": {reason: reason_counts[reason] for reason in all_reasons},
                "by_role": {
                    role: {reason: reason_counts_by_role[role][reason] for reason in all_reasons}
                    for role in ROLES
                },
                "games": total,
            },
            "average_rounds": {
                "overall": _avg(rounds_all),
                "by_role": {role: _avg(rounds_by_role[role]) for role in ROLES},
            },
        }
    }


def compute_gsir(games: list[dict[str, Any]]) -> dict[str, Any]:
    actions_by_role: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}
    cumulative_by_role: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}

    for game in games:
        alice_id, alice_role = _find_alice(game)
        if alice_id is None or alice_role not in actions_by_role:
            continue

        logs = game.get("logs", [])
        if not isinstance(logs, list) or len(logs) < 2:
            continue

        game_cumulative = 0.0
        game_has_action = False
        for idx, log in enumerate(logs[:-1]):
            president_id = log.get("presidentId")
            chancellor_id = log.get("chancellorId")
            if president_id != alice_id and chancellor_id != alice_id:
                continue

            current_score = log.get("gameStateScore", 0.0)
            next_score = logs[idx + 1].get("gameStateScore", 0.0)
            delta = float(next_score) - float(current_score)
            if alice_role != "liberal":
                delta = -delta

            actions_by_role[alice_role].append(delta)
            game_cumulative += delta
            game_has_action = True

        if game_has_action:
            cumulative_by_role[alice_role].append(game_cumulative)

    def _mean(values: list[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    cumulative_role_mean = {role: _mean(vals) * 100 for role, vals in cumulative_by_role.items()}
    return {
        "gsir": {
            "overall": _mean(
                cumulative_by_role["liberal"]
                + cumulative_by_role["fascist"]
                + cumulative_by_role["hitler"]
            )
            * 100,
            "by_role": cumulative_role_mean,
            "actions": {role: len(vals) for role, vals in actions_by_role.items()},
        }
    }


def compute_ria(games: list[dict[str, Any]]) -> dict[str, Any]:
    own_role_scores: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}
    target_role_scores: dict[str, list[float]] = {"liberal": [], "fascist": [], "hitler": []}
    liberal_total_score = 0.0
    liberal_total_count = 0

    for game in games:
        players = game.get("players", [])
        if not isinstance(players, list) or not players:
            continue

        roles = {p.get("username"): p.get("role", "").lower() for p in players}
        alice_id, alice_role = _find_alice(game)
        if alice_id is None or alice_role not in {"liberal", "fascist", "hitler"}:
            continue
        alice_name = players[alice_id].get("username")
        if not isinstance(alice_name, str):
            continue

        logs = game.get("logs", [])
        if not isinstance(logs, list):
            continue

        for log in logs:
            rapid = log.get("rapidAssessments")
            if not isinstance(rapid, dict):
                continue
            assessment_text = rapid.get(str(alice_id))
            if not isinstance(assessment_text, str):
                continue

            parsed = _parse_rapid_assessment(
                assessment_text,
                [name for name in roles if name != alice_name],
                keep_unknown=False,
            )
            for target_name, guessed in parsed.items():
                true_role = roles.get(target_name)
                if true_role not in {"liberal", "fascist", "hitler"}:
                    continue
                score = _belief_score(guessed, true_role)
                own_role_scores[alice_role].append(score)

                if alice_role == "liberal":
                    target_role_scores[true_role].append(score)
                    liberal_total_score += score
                    liberal_total_count += 1

    def _pct(scores: list[float]) -> float:
        if not scores:
            return 0.0
        return float(sum(scores)) / len(scores) * 100.0

    return {
        "ria": {
            "overall_liberal_alice": _safe_pct(liberal_total_score, liberal_total_count),
            "by_own_role": {role: _pct(scores) for role, scores in own_role_scores.items()},
            "by_target_role": {role: _pct(scores) for role, scores in target_role_scores.items()},
            "samples_liberal_alice": liberal_total_count,
        }
    }


def compute_drr(games: list[dict[str, Any]]) -> dict[str, Any]:
    round_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"success": 0, "half": 0, "failure": 0}
    )

    for game in games:
        players = game.get("players", [])
        if not isinstance(players, list) or not players:
            continue

        alice_id, alice_role = _find_alice(game)
        if alice_id is None or alice_role not in {"fascist", "hitler"}:
            continue
        alice_name = players[alice_id].get("username")
        if not isinstance(alice_name, str):
            continue

        player_roles = {idx: p.get("role", "").lower() for idx, p in enumerate(players)}
        logs = game.get("logs", [])
        if not isinstance(logs, list):
            continue

        for round_idx, log in enumerate(logs, start=1):
            rapid = log.get("rapidAssessments")
            if not isinstance(rapid, dict):
                continue

            for pid_str, assessment_text in rapid.items():
                try:
                    pid = int(pid_str)
                except (TypeError, ValueError):
                    continue
                if pid == alice_id:
                    continue
                if player_roles.get(pid) != "liberal":
                    continue

                parsed = _parse_rapid_assessment(
                    str(assessment_text),
                    [alice_name],
                    keep_unknown=True,
                )
                perceived = parsed.get(alice_name, "unknown").lower()

                if perceived in {"liberal", "unknown"}:
                    round_stats[round_idx]["success"] += 1
                elif perceived == alice_role:
                    round_stats[round_idx]["failure"] += 1
                elif perceived in {"fascist", "hitler"}:
                    round_stats[round_idx]["half"] += 1
                else:
                    round_stats[round_idx]["success"] += 1

    by_round: dict[int, float] = {}
    for round_idx in sorted(round_stats.keys()):
        stats = round_stats[round_idx]
        total = stats["success"] + stats["half"] + stats["failure"]
        if total == 0:
            continue
        by_round[round_idx] = ((stats["success"] + 0.5 * stats["half"]) / total) * 100

    average = (sum(by_round.values()) / len(by_round)) if by_round else 0.0
    return {
        "drr": {
            "average": average,
            "by_round": by_round,
            "rounds_with_data": len(by_round),
        }
    }


def compute_voting_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _empty_vote_bucket()
    by_role = {role: _empty_vote_bucket() for role in ROLES}

    for game in games:
        players = game.get("players", [])
        logs = game.get("logs", [])
        alice_id, alice_role = _find_alice(game)
        if (
            alice_id is None
            or alice_role not in ROLES
            or not isinstance(players, list)
            or not isinstance(logs, list)
        ):
            continue

        player_roles = {idx: p.get("role", "").lower() for idx, p in enumerate(players)}
        fascist_policies_before_vote = 0

        for round_idx, log in enumerate(logs, start=1):
            votes = log.get("votes")
            if isinstance(votes, list) and alice_id < len(votes):
                vote_value = votes[alice_id]
                if isinstance(vote_value, bool):
                    phase = _phase_for_round(round_idx)
                    voted_ja = bool(vote_value)
                    _record_vote(overall, voted_ja, phase)
                    _record_vote(by_role[alice_role], voted_ja, phase)

                    dangerous = False
                    chancellor_id = log.get("chancellorId")
                    if isinstance(chancellor_id, int) and fascist_policies_before_vote >= 3:
                        chancellor_role = player_roles.get(chancellor_id)
                        dangerous = chancellor_role in {"fascist", "hitler"}

                    if dangerous:
                        _record_danger_vote(overall, voted_ja)
                        _record_danger_vote(by_role[alice_role], voted_ja)

            enacted_policy = str(log.get("enactedPolicy", "")).lower()
            if enacted_policy == "fascist":
                fascist_policies_before_vote += 1

    approval = _vote_bucket_report(overall)
    approval_by_role = {role: _vote_bucket_report(bucket) for role, bucket in by_role.items()}

    accuracy = _danger_bucket_report(overall, key="vote_accuracy")
    accuracy_by_role = {
        role: _danger_bucket_report(bucket, key="vote_accuracy") for role, bucket in by_role.items()
    }

    danger_approval = _danger_bucket_report(overall, key="danger_approval")
    danger_approval_by_role = {
        role: _danger_bucket_report(bucket, key="danger_approval") for role, bucket in by_role.items()
    }

    return {
        "voting": {
            "approval_rate": {
                **approval,
                "by_own_role": {role: data["overall"] for role, data in approval_by_role.items()},
                "by_own_role_full": approval_by_role,
            },
            "vote_accuracy": {
                **accuracy,
                "by_own_role": {role: data["overall"] for role, data in accuracy_by_role.items()},
                "by_own_role_full": accuracy_by_role,
            },
            "dangerous_approval_rate": {
                **danger_approval,
                "by_own_role": {
                    role: data["overall"] for role, data in danger_approval_by_role.items()
                },
                "by_own_role_full": danger_approval_by_role,
            },
        }
    }


def compute_endorsement_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"yes": 0, "total": 0, "nominations": 0}
    by_role: dict[str, dict[str, int]] = {
        role: {"yes": 0, "total": 0, "nominations": 0} for role in ROLES
    }

    for game in games:
        players = game.get("players", [])
        logs = game.get("logs", [])
        alice_id, alice_role = _find_alice(game)
        if (
            alice_id is None
            or alice_role not in ROLES
            or not isinstance(players, list)
            or not isinstance(logs, list)
        ):
            continue

        for log in logs:
            president_id = log.get("presidentId")
            if president_id != alice_id:
                continue

            votes = log.get("votes")
            if not isinstance(votes, list):
                continue

            other_votes = [
                vote
                for idx, vote in enumerate(votes)
                if idx != alice_id and isinstance(vote, bool)
            ]
            if not other_votes:
                continue

            yes_count = sum(1 for vote in other_votes if vote)
            total_count = len(other_votes)

            counts["nominations"] += 1
            counts["yes"] += yes_count
            counts["total"] += total_count

            by_role[alice_role]["nominations"] += 1
            by_role[alice_role]["yes"] += yes_count
            by_role[alice_role]["total"] += total_count

    def _role_rate(data: dict[str, int]) -> float:
        return _safe_pct(data["yes"], data["total"])

    return {
        "endorsement_rate": {
            "overall": _safe_pct(counts["yes"], counts["total"]),
            "by_own_role": {role: _role_rate(data) for role, data in by_role.items()},
            "counts": {
                "overall": counts,
                "by_own_role": by_role,
            },
        }
    }


def compute_agreeableness_metrics(voting_metrics: dict[str, Any]) -> dict[str, Any]:
    voting = voting_metrics.get("voting", {})
    approval = voting.get("approval_rate", {})
    approval_by_phase = approval.get("by_phase", {})
    danger_approval = voting.get("dangerous_approval_rate", {})

    early = float(approval_by_phase.get("early", 0.0))
    late = float(approval_by_phase.get("late", 0.0))
    overall = float(approval.get("overall", 0.0))
    danger = float(danger_approval.get("overall", 0.0))

    return {
        "agreeableness": {
            "overall_ja_rate": overall,
            "late_game_ja_rate": late,
            "dangerous_government_ja_rate": danger,
            "early_to_late_drop": early - late,
            "dangerous_minus_overall": danger - overall,
        }
    }


def compute_benchmark_metrics(games: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(compute_win_rates(games))
    metrics.update(compute_game_outcomes(games))
    metrics.update(compute_gsir(games))
    metrics.update(compute_ria(games))
    metrics.update(compute_drr(games))

    voting_metrics = compute_voting_metrics(games)
    metrics.update(voting_metrics)
    metrics.update(compute_endorsement_metrics(games))
    metrics.update(compute_agreeableness_metrics(voting_metrics))
    return metrics
