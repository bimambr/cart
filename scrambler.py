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
from typing import Literal, TypedDict, cast

from _types import State, TranslationAttempt


class Entry(TypedDict):
    text_id: int
    source_text: str
    translation: str


class ScrambledRow(TypedDict):
    pair_id: int
    target_idiom: str
    source_text: str
    translation_A: str
    translation_B: str
    translation_C: str
    translation_D: str


class KeyEntry(TypedDict):
    text_id: int
    mapping: dict[Literal["A", "B", "C", "D"], Literal["T1", "T2", "T3", "T4"]]


@dataclass
class _CLIArgs:
    T13: str
    T24: str
    idioms: str
    out_csv: str
    out_key: str
    override: bool


def load_jsonl_translations(filepath: str, grab_last: bool = False) -> dict[int, Entry]:
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
                translation = (attempts or [{}])[-1 if grab_last else 0].get(
                    "translation"
                ) or "N/A"
                data[text_id] = {
                    "text_id": text_id,
                    "source_text": source_entry.get("text", ""),
                    "translation": translation,
                }
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Scramble the conditioned translations for blind ABX testing."
    )
    _ = parser.add_argument(
        "T13",
        help="Path to T13.jsonl (non-RAG refinement stack)",
    )
    _ = parser.add_argument(
        "T24",
        help="Path to T24.jsonl (RAG refinement stack)",
    )
    _ = parser.add_argument(
        "idioms",
        help="Path to idioms.json for row decoupling",
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

    # T1 & T3 and T2 & T4 are generated sequentially
    # T1 == T13[0] and T3 == T13[-1]
    # T2 == T24[0] and T4 == T24[-1]
    t1_map = load_jsonl_translations(args.T13)
    t3_map = load_jsonl_translations(args.T13, grab_last=True)
    t2_map = load_jsonl_translations(args.T24)
    t4_map = load_jsonl_translations(args.T24, grab_last=True)

    common_ids = sorted(list(set(t1_map.keys()) & set(t2_map.keys())))
    if not common_ids:
        print("Error: No matching text IDs found between files.", file=sys.stderr)
        sys.exit(1)

    with open(args.idioms, "r") as f:
        idioms: list[str] = cast("list[str]", json.load(f))

    scrambled_rows: list[ScrambledRow] = []
    key_mapping = {}
    pair_counter = 1

    for text_id in common_ids:
        items = {
            "T1": t1_map[text_id]["translation"],
            "T2": t2_map[text_id]["translation"],
            "T3": t3_map[text_id]["translation"],
            "T4": t4_map[text_id]["translation"],
        }

        lowered = t1_map[text_id]["source_text"].lower()
        detected_idioms = [i for i in idioms if i.lower() in lowered]

        if not detected_idioms:
            print(f"Warning: no idioms detected for text {text_id}. Skipping...")
            continue

        for idiom in detected_idioms:
            labels = ["T1", "T2", "T3", "T4"]
            random.shuffle(labels)
            row_mapping = {
                col: label for col, label in zip(["A", "B", "C", "D"], labels)
            }
            scrambled_rows.append(
                ScrambledRow(
                    pair_id=pair_counter,
                    target_idiom=idiom,
                    source_text=t1_map[text_id]["source_text"],
                    translation_A=items[row_mapping["A"]],
                    translation_B=items[row_mapping["B"]],
                    translation_C=items[row_mapping["C"]],
                    translation_D=items[row_mapping["D"]],
                )
            )
            key_mapping[str(pair_counter)] = {
                "text_id": text_id,
                "mapping": row_mapping,
            }
            pair_counter += 1

    random.shuffle(scrambled_rows)

    headers = [
        "pair_id",
        "source_text",
        "target_idiom",
        "translation_A",
        "translation_B",
        "translation_C",
        "translation_D",
        "accuracy_A",
        "acceptability_A",
        "readability_A",
        "accuracy_B",
        "acceptability_B",
        "readability_B",
        "accuracy_C",
        "acceptability_C",
        "readability_C",
        "accuracy_D",
        "acceptability_D",
        "readability_D",
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
                    "accuracy_C": "",
                    "acceptability_C": "",
                    "readability_C": "",
                    "accuracy_D": "",
                    "acceptability_D": "",
                    "readability_D": "",
                }
            )

    with open(args.out_key, "w", encoding="utf-8") as f:
        json.dump(key_mapping, f, indent=4)

    print("Files written to", args.out_csv, "and", args.out_key)


if __name__ == "__main__":
    main()
