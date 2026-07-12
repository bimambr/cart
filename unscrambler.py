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
import json
import sys
from dataclasses import dataclass
from typing import TypedDict, cast

from scrambler import KeyEntry


TREATMENT_META = {
    "T1": {"rag_status": "RAG-", "refine_status": "Refine-"},
    "T2": {"rag_status": "RAG+", "refine_status": "Refine-"},
    "T3": {"rag_status": "RAG-", "refine_status": "Refine+"},
    "T4": {"rag_status": "RAG+", "refine_status": "Refine+"},
}


@dataclass
class _CLIArgs:
    evaluated_csv: str
    key_json: str
    out: str


class LongRow(TypedDict):
    idiom_id: int
    source_text: str
    target_idiom: str
    treatment: str
    rag_status: str
    refine_status: str
    translation: str
    accuracy: float
    acceptability: float
    readability: float
    weighted_tqa: float


def calculate_tqa(acc: float, accp: float, read: float) -> float:
    return ((acc * 3) + (accp * 2) + (read * 1)) / 6


def main():
    parser = argparse.ArgumentParser(
        description="Decode evaluation matrix and pivot to long format for R analysis."
    )
    _ = parser.add_argument("evaluated_csv", help="Path to the scored blind_test.csv")
    _ = parser.add_argument("key_json", help="Path to the blind_key.json mapping key")
    _ = parser.add_argument(
        "-o",
        "--out",
        default="translations_long.csv",
        help="Path to save the unscrambled long-format dataset",
    )
    args = parser.parse_args(namespace=_CLIArgs)

    with open(args.key_json, "r", encoding="utf-8") as f:
        key_mapping = cast("dict[str, KeyEntry]", json.load(f))

    long_rows: list[LongRow] = []

    with open(args.evaluated_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pair_id"]
            if pid not in key_mapping:
                continue

            mapping = key_mapping[pid]["mapping"]

            try:
                scores = {
                    "A": {
                        "acc": float(row["accuracy_A"]),
                        "accp": float(row["acceptability_A"]),
                        "read": float(row["readability_A"]),
                    },
                    "B": {
                        "acc": float(row["accuracy_B"]),
                        "accp": float(row["acceptability_B"]),
                        "read": float(row["readability_B"]),
                    },
                    "C": {
                        "acc": float(row["accuracy_C"]),
                        "accp": float(row["acceptability_C"]),
                        "read": float(row["readability_C"]),
                    },
                    "D": {
                        "acc": float(row["accuracy_D"]),
                        "accp": float(row["acceptability_D"]),
                        "read": float(row["readability_D"]),
                    },
                }
            except (ValueError, KeyError, TypeError):
                print(
                    f"Warning: Missing or malformed data at pair_id {pid}. Skipping row.",
                    file=sys.stderr,
                )
                continue

            for col, tx in mapping.items():
                acc = scores[col]["acc"]
                accp = scores[col]["accp"]
                read = scores[col]["read"]
                tqa = calculate_tqa(acc, accp, read)

                long_rows.append(
                    LongRow(
                        idiom_id=int(pid),
                        source_text=row["source_text"],
                        target_idiom=row["target_idiom"],
                        treatment=tx,
                        rag_status=TREATMENT_META[tx]["rag_status"],
                        refine_status=TREATMENT_META[tx]["refine_status"],
                        translation=row[f"translation_{col}"],
                        accuracy=acc,
                        acceptability=accp,
                        readability=read,
                        weighted_tqa=tqa,
                    )
                )

    long_headers = [
        "idiom_id",
        "source_text",
        "target_idiom",
        "treatment",
        "rag_status",
        "refine_status",
        "translation",
        "accuracy",
        "acceptability",
        "readability",
        "weighted_tqa",
    ]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_headers)
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"Data successfully unscrambled and pivoted to long format: {args.out}")


if __name__ == "__main__":
    main()
