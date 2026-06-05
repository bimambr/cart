"""
Copyright 2026 Muhammad Bima Ramadhan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import os
import random
import sys
from typing import TypedDict, cast

from _types import State, TranslationAttempt


class Entry(TypedDict):
    text_id: int
    source_text: str
    translation: str


class ScrambledRow(TypedDict):
    pair_id: int
    source_text: str
    translation_A: str
    translation_B: str


class KeyEntry(TypedDict):
    text_id: int
    translation_A: str
    translation_B: str


@dataclass
class _CLIArgs:
    baseline: str
    ce: str
    out_csv: str
    out_key: str
    override: bool


def load_jsonl_translations(filepath: str) -> dict[int, Entry]:
    data: dict[int, Entry] = {}
    decoder = json.JSONDecoder()
    buffer = ""

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            buffer = buffer.lstrip()

            if not buffer:
                continue

            try:
                state, idx = cast("tuple[State, int]", decoder.raw_decode(buffer))
                buffer = buffer[idx:]
            except json.JSONDecodeError:
                continue
            else:
                source_entry = state.get("source_text", {})
                text_id = source_entry.get("id")

                assert text_id

                history = state.get("history", [])
                attempts = [
                    cast(TranslationAttempt, h)
                    for h in history
                    if h.get("type") == "attempt"
                ]
                translation = (attempts or [{}])[-1].get("translation") or "N/A"
                data[text_id] = {
                    "text_id": text_id,
                    "source_text": source_entry.get("text", ""),
                    "translation": translation,
                }
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Scramble baseline and CE translations for blind ABX testing."
    )
    _ = parser.add_argument(
        "baseline",
        help="Path to baseline.jsonl",
    )
    _ = parser.add_argument(
        "ce",
        help="Path to context_engineered.jsonl",
    )
    _ = parser.add_argument(
        "--out_csv",
        default="blind_test.csv",
        help="Output evaluation CSV",
    )
    _ = parser.add_argument(
        "--out_key",
        default="blind_key.json",
        help="Output decoding key mapping",
    )
    _ = parser.add_argument(
        "-y",
        action="store_true",
        dest="override",
        default=False,
        help="Override existing files",
    )
    args = parser.parse_args(namespace=_CLIArgs)

    base_map = load_jsonl_translations(args.baseline)
    ce_map = load_jsonl_translations(args.ce)

    common_ids = sorted(list(set(base_map.keys()) & set(ce_map.keys())))
    if not common_ids:
        print("Error: No matching text IDs found between files.", file=sys.stderr)
        sys.exit(1)

    scrambled_rows: list[ScrambledRow] = []
    key_mapping = {}

    for pair_id, text_id in enumerate(common_ids, start=1):
        b_item = base_map[text_id]
        c_item = ce_map[text_id]

        flip = random.choice([True, False])
        if flip:
            trans_a, label_a = c_item["translation"], "ce"
            trans_b, label_b = b_item["translation"], "baseline"
        else:
            trans_a, label_a = b_item["translation"], "baseline"
            trans_b, label_b = c_item["translation"], "ce"

        scrambled_rows.append(
            {
                "pair_id": pair_id,
                "source_text": b_item["source_text"],
                "translation_A": trans_a,
                "translation_B": trans_b,
            }
        )

        key_mapping[str(pair_id)] = {
            "text_id": text_id,
            "translation_A": label_a,
            "translation_B": label_b,
        }

    random.shuffle(scrambled_rows)

    headers = [
        "pair_id",
        "source_text",
        "translation_A",
        "translation_B",
        "accuracy_A",
        "acceptability_A",
        "readability_A",
        "accuracy_B",
        "acceptability_B",
        "readability_B",
    ]

    if (
        os.path.exists(args.out_csv) or os.path.exists(args.out_key)
    ) and not args.override:
        print("Error: file conflicts. Use `-y` to override.", file=sys.stderr)
        sys.exit(1)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in scrambled_rows:
            writer.writerow(
                {
                    **row,
                    "accuracy_A": "",
                    "acceptability_A": "",
                    "readability_A": "",
                    "accuracy_B": "",
                    "acceptability_B": "",
                    "readability_B": "",
                }
            )

    with open(args.out_key, "w", encoding="utf-8") as f:
        json.dump(key_mapping, f, indent=4)

    print("Files written to", args.out_csv, "and", args.out_key)


if __name__ == "__main__":
    main()
