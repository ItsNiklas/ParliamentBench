"""Rules check: play many games of random players at every table size and verify the engine
follows the official Secret Hitler rules. No model calls; runs in a few minutes.

    python -m tests.test_rules            # 150 games per table size (5-10 players)
    python -m tests.test_rules --games 500 --seed 7

Each check below quotes the rule it tests (the official rulebook, secrethitler.com).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import tempfile
from collections import Counter
from types import SimpleNamespace

from parliamentbench.config_loader import Config
from parliamentbench.game_engine import ParliamentGame
from parliamentbench.game_primitives import PLAYERS
from parliamentbench.game_state import GameState
from parliamentbench.players import RandomPlayer

PEEKS: list = []
VETO_CALLS: list = []
_orig_veto = RandomPlayer.veto


def _view(self, policies):
    PEEKS.append([p.type for p in policies])


def _veto(self, policies):
    seat = "chancellor" if self is self.state.chancellor else "president"
    decision = _orig_veto(self, policies)
    VETO_CALLS.append((seat, decision, self.state.veto_proposed_by is not None))
    return decision


RESHUFFLES: list = []  # turn numbers at which the official end-of-session reshuffle happened
_orig_reshuffle = GameState.reshuffle_if_low


def _reshuffle(self):
    if len(self.policies) < 3:
        RESHUFFLES.append(self.turn_count)
    return _orig_reshuffle(self)


RandomPlayer.view_policies = _view
RandomPlayer.veto = _veto
GameState.reshuffle_if_low = _reshuffle


class CheckedGame(ParliamentGame):
    """The engine as is, with a record of what happened each turn."""

    def turn(self):
        rec = {"alive": [p for p in self.state.players if not p.is_dead], "last_elected": self.last_elected,
               "chat_before": len(self.state.chat_log), "power": None, "peek_index": None, "veto_calls": len(VETO_CALLS)}
        self.records.append(rec)
        done = super().turn()
        rec["president"] = self.state.president
        rec["nominee"] = self.state.chancellor
        rec["chat_after"] = len(self.state.chat_log)
        rec["entry"] = self.state.game_data_logs[-1] if self.state.game_data_logs else None
        rec["done"] = done
        return done

    def perform_vote_action(self):
        action = self.state.fascist_track_actions[self.state.fascist_track - 1]
        rec = self.records[-1]
        rec["power"] = action
        rec["power_president"] = self.state.president
        if action == "policy":
            rec["peek_index"] = len(PEEKS)
        super().perform_vote_action()
        if action == "choose":
            rec["special"] = (self.state.president, self.special_president)


def _passed(e):
    v = e.get("votes") or []
    return sum(x is True for x in v) > sum(x is False for x in v)


def play(n_players: int, out: str) -> CheckedGame:
    cfg = Config()
    cfg.player_types = ["RANDOM"] * n_players
    cfg.llm["random_player"] = {"api_key": "unused", "base_url": "http://127.0.0.1:9/v1", "model": "none"}
    args = SimpleNamespace(config=None, players=n_players, output=out + "/log", gamestate_json=None, chat_json=None,
                           cutoff_rounds=0, log_level="ERROR", simple=False, summary_path=out,
                           player_types=cfg.player_types, role=None)
    game = CheckedGame(args, config=cfg)
    game.records = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        game.play()
    return game


def check_game(g: CheckedGame, stats: Counter, fail) -> None:
    recs = g.records
    players = g.state.players
    in_a_row = 0
    for i, r in enumerate(recs):
        e = r["entry"] or {}
        passed = _passed(e) if e else False
        enacted = e.get("enactedPolicy")
        blind = enacted in ("liberal", "fascist") and (not passed or e.get("vetoUsed"))
        # Term limits — "The last elected President and Chancellor are term-limited … If there are only
        # five players left in the game, only the last elected Chancellor is ineligible."
        last_pres, last_chan = r["last_elected"]
        nominee = players[e["chancellorId"]] if e.get("chancellorId") is not None else None
        if nominee is not None:
            stats["nominations"] += 1
            if nominee is last_chan:
                fail("nominated the last elected Chancellor")
            if len(r["alive"]) > 5 and nominee is last_pres:
                fail("nominated the last elected President with more than five players alive")
            if len(r["alive"]) <= 5 and nominee is last_pres:
                stats["last_president_nominated_at_5"] += 1
            if i and recs[i - 1]["entry"] and not _passed(recs[i - 1]["entry"]) and nominee is recs[i - 1]["nominee"]:
                stats["rejected_nominee_renominated"] += 1
        # Election tracker — "If three governments are rejected in a row … the top Policy is enacted";
        # "Any time a new Policy tile is played face-up, the Election Tracker is reset."
        if passed and enacted in ("liberal", "fascist") and not e.get("vetoUsed"):
            in_a_row = 0
        elif e and not (passed and enacted is None):
            in_a_row += 1
            if blind:
                stats["blind_policies"] += 1
                if in_a_row != 3:
                    fail(f"blind policy after {in_a_row} failed elections in a row")
                in_a_row = 0
        # Chaos — "Any power granted by this Policy is ignored."
        if blind and r["power"]:
            fail("presidential power after a blind policy")
        # Discussion after every enacted policy that doesn't end the game.
        if enacted in ("liberal", "fascist") and not r["done"]:
            stats["policies"] += 1
            after = [c for c in g.state.chat_log[r["chat_before"]:r["chat_after"]] if c.get("state") == "after_policy"]
            if not after:
                fail("no discussion after an enacted policy")
        # Policy Peek — "looks at the top three tiles … returns them to the top of the deck without
        # changing the order": the next three cards drawn are the peeked ones.
        if r["peek_index"] is not None:
            peeked = PEEKS[r["peek_index"]]
            drawn, last_turn = [], i + 1
            for j, later in enumerate(recs[i + 1:], start=i + 2):
                le = later["entry"] or {}
                if le.get("presidentHand"):
                    drawn += le["presidentHand"]
                    if le.get("vetoUsed") and le.get("enactedPolicy") in ("liberal", "fascist"):
                        drawn.append(le["enactedPolicy"])
                elif le and not _passed(le) and le.get("enactedPolicy") in ("liberal", "fascist"):
                    drawn.append(le["enactedPolicy"])
                last_turn = j
                if len(drawn) >= 3:
                    break
            if len(drawn) >= 3:
                stats["peeks_checked"] += 1
                if drawn[:3] != peeked:
                    # Only legal when a blind policy took a peeked card and the deck, down to under
                    # three, was reshuffled (official end-of-session rule) before the next draw.
                    if any(i + 1 < t < last_turn + 1 for t in RESHUFFLES):
                        stats["peek_mismatch_after_reshuffle"] += 1
                    else:
                        fail("peek: the next cards drawn are not the peeked ones")
        # Special Election — the chosen player presides next; then the placard returns to the left of
        # the President who called it.
        if r.get("special") and i + 2 < len(recs) + 1:
            caller, chosen = r["special"]
            if i + 1 < len(recs):
                stats["special_elections"] += 1
                if recs[i + 1]["president"] is not chosen:
                    fail("special election: the chosen player did not preside next")
                if i + 2 < len(recs):
                    alive = recs[i + 2]["alive"]
                    seat = caller.id
                    while True:
                        seat = (seat + 1) % len(players)
                        if players[seat] in alive or players[seat] is caller:
                            break
                    if recs[i + 2]["president"] is not players[seat] and players[seat] in alive:
                        fail("special election: rotation did not resume left of the caller")
        # Veto — the President is only asked after the Chancellor proposes.
        calls = VETO_CALLS[r["veto_calls"]:recs[i + 1]["veto_calls"] if i + 1 < len(recs) else len(VETO_CALLS)]
        for k, (seat, decision, proposed) in enumerate(calls):
            if seat == "president":
                stats["veto_consents_asked"] += 1
                if not proposed or k == 0 or calls[k - 1] != ("chancellor", True, False):
                    fail("veto: President asked without a Chancellor proposal")
    # Investigations — "No player may be investigated twice in the same game."
    if len(g.state.investigated) != len(set(map(id, g.state.investigated))):
        fail("a player was investigated twice")
    for p in players:
        stats["investigations"] += len(p.investigations)
        if p.investigations and "investigations" not in p.get_known_state():
            fail("an investigator's own result is missing from its state")
    # Game end
    if g.game_end_reason == "no_clear_winner":
        fail("game hit the round cap")
    if g.game_end_reason == "hitler_chancellor" and not _passed(g.state.game_data_logs[-1]):
        fail("'Hitler elected' scored without an election")
    # Deck — 17 policies at all times
    total = len(g.state.policies) + len(g.state.discards) + g.state.liberal_track + g.state.fascist_track
    if total != 17:
        fail(f"deck has {total} policies")
    stats[g.game_end_reason] += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=150, help="games per table size")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    if a.seed is not None:
        random.seed(a.seed)
    failures: Counter = Counter()
    out = tempfile.mkdtemp()
    for n in sorted(PLAYERS):
        stats: Counter = Counter()
        for _ in range(a.games):
            PEEKS.clear()
            VETO_CALLS.clear()
            RESHUFFLES.clear()
            check_game(play(n, out), stats, lambda msg: failures.update([f"{n}p: {msg}"]))
        print(f"{n} players: {a.games} games · endings "
              + ", ".join(f"{k} {stats[k]}" for k in ("liberal_policy", "hitler_killed", "fascist_policy", "hitler_chancellor"))
              + f" · blind policies {stats['blind_policies']} · peeks checked {stats['peeks_checked']}"
              f" (mismatch after a reshuffle {stats['peek_mismatch_after_reshuffle']})"
              f" · special elections {stats['special_elections']} · investigations {stats['investigations']}"
              f" · veto consents {stats['veto_consents_asked']} · rejected nominees renominated {stats['rejected_nominee_renominated']}")
    if failures:
        print("\nRULE VIOLATIONS:")
        for msg, k in failures.most_common():
            print(f"  {k}× {msg}")
        return 1
    print("\nAll rules checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
