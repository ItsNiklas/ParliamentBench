"""Post-run verifier for the neutral arm.

Scans every game's chat log + log entries + reflections in
`runsF2-LLAMA3370B-Xneut/*_summary.json` for any leak of the forbidden
loaded-term tokens ("Hitler", "Secret Hitler", and their lowercase /
UPPERCASE variants).

This catches two kinds of failure:
  1. A new prompt path (e.g., new LLM call site added later) bypassed
     parliamentbench.theme.apply_to_messages and the LLM saw a loaded term.
  2. The model itself generated a forbidden token in its chat output
     (it shouldn't, given it never saw the term, but worth checking).

A clean run produces zero hits. Non-zero hits print the file, the field,
and a 60-char context window.

Usage:
    python analysis/verify_neutral_clean.py runsF2-LLAMA3370B-Xneut
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORBIDDEN = ("Hitler", "Secret Hitler", "hitler", "secret hitler", "HITLER")


def scan_value(value, path: str, hits: list):
    if isinstance(value, str):
        for tok in FORBIDDEN:
            i = value.find(tok)
            if i >= 0:
                start = max(0, i - 30)
                end = min(len(value), i + len(tok) + 30)
                hits.append((path, tok, value[start:end]))
                return  # one hit per field is enough
    elif isinstance(value, dict):
        for k, v in value.items():
            scan_value(v, f"{path}.{k}", hits)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            scan_value(v, f"{path}[{i}]", hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="runs* folder to scan (typically the neutral arm)")
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"not a directory: {folder}")

    files = sorted(folder.glob("*_summary.json"))
    if not files:
        sys.exit(f"no summary files in {folder}")

    # Fields the LLM-bound substitution should have caught. The simulator's
    # *internal* game_log, chat_log, and players (role labels) all end up
    # echoed into the next prompt, so any leak there is a substitution miss.
    SCAN_FIELDS = ("chats", "logs")

    total_hits = 0
    files_with_hits = 0
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        file_hits: list = []
        for field in SCAN_FIELDS:
            scan_value(data.get(field), f"{fp.name}.{field}", file_hits)
        if file_hits:
            files_with_hits += 1
            total_hits += len(file_hits)
            print(f"\n=== {fp.name} ===")
            for path, tok, ctx in file_hits[:10]:
                print(f"  {path}  ({tok!r})  ...{ctx!r}...")

    print("\n" + "=" * 70)
    print(f"Scanned: {len(files)} files")
    print(f"Files with leaks: {files_with_hits}")
    print(f"Total forbidden-token hits: {total_hits}")
    if total_hits == 0:
        print("\nVerdict: CLEAN. No forbidden tokens reached the data.")
    else:
        print("\nVerdict: LEAK. Inspect the listed paths; either the LLM was")
        print("         actually fed a loaded term, or the LLM hallucinated it")
        print("         on its own. The latter is itself worth reporting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
