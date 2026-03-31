"""ParliamentBench benchmark runner CLI."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .game_engine import ParliamentGame
from .config_loader import Config
from .metric.benchmark_metrics import compute_benchmark_metrics
from .runtime import apply_runtime_settings


def _load_config(path: str | None) -> Config:
    if path:
        return Config.from_yaml(path)
    return Config()


def _role_schedule(
    num_games: int,
    schedule: str,
    fixed_role: str,
    rng: random.Random,
) -> list[str]:
    if schedule == "fixed":
        return [fixed_role] * num_games
    if schedule == "random":
        roles = ["liberal", "fascist", "hitler"]
        return [rng.choice(roles) for _ in range(num_games)]

    # balanced: 60/20/20 pattern over each 5 games, then truncated.
    pattern = ["liberal", "liberal", "liberal", "fascist", "hitler"]
    result: list[str] = []
    while len(result) < num_games:
        result.extend(pattern)
    return result[:num_games]


def _build_game_args(
    config: Config,
    role: str,
    summary_path: Path,
    log_level: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=None,
        players=config.players,
        output=str(summary_path / "benchmark.log"),
        gamestate_json=None,
        chat_json=None,
        cutoff_rounds=config.cutoff_rounds,
        log_level=log_level,
        simple=False,
        summary_path=str(summary_path),
        player_types=config.player_types,
        role=role,
    )


def _collect_game_summaries(summary_path: Path) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for summary_file in sorted(summary_path.glob("*_summary.json")):
        try:
            with summary_file.open("r", encoding="utf-8") as f:
                games.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return games


def _format_phase_map(data: dict[str, float]) -> str:
    return (
        f"early={data.get('early', 0.0):.1f}% | "
        f"mid={data.get('mid', 0.0):.1f}% | "
        f"late={data.get('late', 0.0):.1f}%"
    )


def _print_summary(metrics: dict[str, Any], num_games: int, output_dir: Path) -> None:
    win_rate = metrics["win_rate"]
    outcomes = metrics["game_outcomes"]
    gsir = metrics["gsir"]
    ria = metrics["ria"]
    drr = metrics["drr"]
    voting = metrics["voting"]
    endorsement = metrics["endorsement_rate"]
    agree = metrics["agreeableness"]

    print("\n" + "=" * 90)
    print("PARLIAMENTBENCH VERBOSE RESULTS")
    print("=" * 90)
    print(f"Games simulated: {num_games}")
    print(f"Output directory: {output_dir}")

    print("\n[1] Win Rates")
    print(
        f"  Overall: {win_rate['overall']:.1f}% (wins={win_rate['counts']['wins']} / games={win_rate['counts']['total']})"
    )
    print(
        "  By role: "
        f"liberal={win_rate['by_role']['liberal']:.1f}% | "
        f"fascist={win_rate['by_role']['fascist']:.1f}% | "
        f"hitler={win_rate['by_role']['hitler']:.1f}%"
    )

    print("\n[2] Game Outcomes")
    print(
        "  Overall ending distribution: "
        f"5L={outcomes['overall']['liberal_policy']:.1f}% | "
        f"Hitler-killed={outcomes['overall']['hitler_killed']:.1f}% | "
        f"6F={outcomes['overall']['fascist_policy']:.1f}% | "
        f"Hitler-chancellor={outcomes['overall']['hitler_chancellor']:.1f}%"
    )
    print(
        "  Avg rounds: "
        f"overall={outcomes['average_rounds']['overall']:.2f} | "
        f"liberal={outcomes['average_rounds']['by_role']['liberal']:.2f} | "
        f"fascist={outcomes['average_rounds']['by_role']['fascist']:.2f} | "
        f"hitler={outcomes['average_rounds']['by_role']['hitler']:.2f}"
    )

    print("\n[3] GSIR")
    print(
        f"  Overall: {gsir['overall']:+.2f} | "
        f"liberal={gsir['by_role']['liberal']:+.2f} | "
        f"fascist={gsir['by_role']['fascist']:+.2f} | "
        f"hitler={gsir['by_role']['hitler']:+.2f}"
    )
    print(
        "  Action counts: "
        f"liberal={gsir['actions']['liberal']} | "
        f"fascist={gsir['actions']['fascist']} | "
        f"hitler={gsir['actions']['hitler']}"
    )

    print("\n[4] RIA")
    print(
        f"  Liberal-Alice overall: {ria['overall_liberal_alice']:.1f}% "
        f"(samples={ria['samples_liberal_alice']})"
    )
    print(
        "  By own role: "
        f"liberal={ria['by_own_role']['liberal']:.1f}% | "
        f"fascist={ria['by_own_role']['fascist']:.1f}% | "
        f"hitler={ria['by_own_role']['hitler']:.1f}%"
    )
    print(
        "  By target role: "
        f"vs-liberal={ria['by_target_role']['liberal']:.1f}% | "
        f"vs-fascist={ria['by_target_role']['fascist']:.1f}% | "
        f"vs-hitler={ria['by_target_role']['hitler']:.1f}%"
    )

    print("\n[5] DRR")
    print(
        f"  Average DRR: {drr['average']:.1f}% | rounds with data={drr['rounds_with_data']}"
    )
    if drr["by_round"]:
        by_round = " | ".join(
            f"r{round_idx}={value:.1f}%" for round_idx, value in drr["by_round"].items()
        )
        print(f"  Round-level: {by_round}")
    else:
        print("  Round-level: no deceptive-role rounds with valid liberal assessments.")

    print("\n[6] Voting Behavior")
    approval = voting["approval_rate"]
    print(
        f"  Approval rate (Ja): overall={approval['overall']:.1f}% | "
        f"{_format_phase_map(approval['by_phase'])}"
    )
    print(
        "  Approval by own role: "
        f"liberal={approval['by_own_role']['liberal']:.1f}% | "
        f"fascist={approval['by_own_role']['fascist']:.1f}% | "
        f"hitler={approval['by_own_role']['hitler']:.1f}%"
    )
    accuracy = voting["vote_accuracy"]
    print(
        f"  Defensive vote accuracy (Nein on dangerous gov): {accuracy['overall']:.1f}% "
        f"(danger_total={accuracy['counts']['danger_total']})"
    )
    print(
        "  Defensive accuracy by own role: "
        f"liberal={accuracy['by_own_role']['liberal']:.1f}% | "
        f"fascist={accuracy['by_own_role']['fascist']:.1f}% | "
        f"hitler={accuracy['by_own_role']['hitler']:.1f}%"
    )
    danger_approval = voting["dangerous_approval_rate"]
    print(
        f"  Dangerous-government Ja rate: {danger_approval['overall']:.1f}% "
        f"(danger_total={danger_approval['counts']['danger_total']})"
    )

    print("\n[7] Endorsement / Agreeableness")
    print(
        f"  Presidential endorsement (received Ja votes): {endorsement['overall']:.1f}% "
        f"(nominations={endorsement['counts']['overall']['nominations']})"
    )
    print(
        "  Endorsement by own role: "
        f"liberal={endorsement['by_own_role']['liberal']:.1f}% | "
        f"fascist={endorsement['by_own_role']['fascist']:.1f}% | "
        f"hitler={endorsement['by_own_role']['hitler']:.1f}%"
    )
    print(
        f"  Agreeableness summary: overall_ja={agree['overall_ja_rate']:.1f}% | "
        f"late_ja={agree['late_game_ja_rate']:.1f}% | "
        f"danger_ja={agree['dangerous_government_ja_rate']:.1f}%"
    )
    print(
        f"  Derived deltas: early_to_late_drop={agree['early_to_late_drop']:.1f}pp | "
        f"danger_minus_overall={agree['dangerous_minus_overall']:.1f}pp"
    )

    print("=" * 90 + "\n")


def run_benchmark(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ParliamentBench Secret-Hitler benchmark simulations.",
    )
    parser.add_argument("--config", "-f", type=str, default=None, help="YAML config path")
    parser.add_argument("--num-games", type=int, default=None, help="Number of simulations")
    parser.add_argument("--players", type=int, default=None, help="Number of players")
    parser.add_argument(
        "--player-types",
        type=lambda s: [part.strip() for part in s.split(",") if part.strip()],
        default=None,
        help="Comma-separated player types (e.g., LLM,CPU,CPU,CPU,CPU)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Override default API key")
    parser.add_argument("--base-url", type=str, default=None, help="Override default API base URL")
    parser.add_argument("--model", type=str, default=None, help="Override default model name")
    parser.add_argument(
        "--role-schedule",
        choices=["balanced", "random", "fixed"],
        default=None,
        help="Role assignment schedule for Alice across games",
    )
    parser.add_argument(
        "--fixed-role",
        choices=["liberal", "fascist", "hitler"],
        default=None,
        help="Role to force when --role-schedule fixed",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for schedule")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summaries and benchmark report files",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run-name suffix for output directory",
    )
    parser.add_argument(
        "--disable-token-summary",
        action="store_true",
        help="Disable per-game token summary printing/saving during benchmark runs",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level for simulator internals",
    )
    parser.add_argument(
        "--theme",
        choices=["original", "neutral"],
        default=None,
        help=(
            "Loaded-terms ablation. 'original' keeps the standard terminology; "
            "'neutral' remaps it to Council/Reds/Blues/Speaker/Deputy before any "
            "text reaches the model"
        ),
    )
    parser.add_argument(
        "--theme-strict",
        action="store_true",
        help="Raise instead of warning when a forbidden term leaks under --theme neutral",
    )
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        default=None,
        help="Request the thinking-ON bundle from reasoning-capable endpoints",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        default=None,
        help="Request the thinking-OFF bundle (reasoning ablation)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high"],
        default=None,
        help="Reasoning effort when reasoning is enabled",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Completion attempts before a player gives up and returns an empty string",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)

    num_games = args.num_games if args.num_games is not None else config.benchmark_num_games
    role_schedule = (
        args.role_schedule if args.role_schedule is not None else config.benchmark_role_schedule
    )
    fixed_role = args.fixed_role if args.fixed_role is not None else config.benchmark_fixed_role
    seed = args.seed if args.seed is not None else config.benchmark_seed
    run_name = args.run_name if args.run_name is not None else config.benchmark_run_name

    if num_games <= 0:
        raise ValueError("--num-games must be > 0")
    if args.players is not None:
        config.players = args.players
    if args.player_types is not None:
        config.player_types = args.player_types
    if args.api_key is not None:
        config.llm["default"]["api_key"] = args.api_key
    if args.base_url is not None:
        config.llm["default"]["base_url"] = args.base_url
    if args.model is not None:
        config.llm["default"]["model"] = args.model
    if args.theme is not None:
        config.theme = args.theme
    if args.theme_strict:
        config.theme_strict = True
    if args.reasoning is not None:
        config.reasoning_enabled = args.reasoning
        # An explicit flag beats the REASONING_ENABLED env var.
        os.environ.pop("REASONING_ENABLED", None)
    if args.reasoning_effort is not None:
        config.reasoning_effort = args.reasoning_effort
    if args.max_retries is not None:
        config.max_retries = args.max_retries

    runtime_settings = apply_runtime_settings(config)
    if runtime_settings["theme"]:
        print(
            f"[theme] active theme = {runtime_settings['theme']} "
            f"(strict={runtime_settings['theme_strict']})"
        )

    rng = random.Random(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_dir = Path("runs") / (
        f"benchmark_{timestamp}" if not run_name else f"benchmark_{run_name}_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.disable_token_summary:
        os.environ["PARLIAMENTBENCH_DISABLE_TOKEN_SUMMARY"] = "1"
    else:
        os.environ.pop("PARLIAMENTBENCH_DISABLE_TOKEN_SUMMARY", None)

    schedule = _role_schedule(num_games, role_schedule, fixed_role, rng)
    for idx, role in enumerate(schedule, start=1):
        print(f"[{idx}/{num_games}] role={role}")
        game_args = _build_game_args(config, role, output_dir, args.log_level)
        game = ParliamentGame(game_args, config=config)
        game.play()

    games = _collect_game_summaries(output_dir)
    metrics = compute_benchmark_metrics(games)
    _print_summary(metrics, num_games=num_games, output_dir=output_dir)

    report_path = output_dir / "benchmark_report_verbose.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "num_games": num_games,
                    "role_schedule": role_schedule,
                    "fixed_role": fixed_role,
                    "seed": seed,
                    "generated_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "runtime_settings": runtime_settings,
                },
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved verbose benchmark report: {report_path}")
    return 0


def main() -> None:
    raise SystemExit(run_benchmark())


if __name__ == "__main__":
    main()
