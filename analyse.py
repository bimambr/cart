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
import sys
from typing import TypedDict, cast
import numpy as np
from scipy.stats import wilcoxon  # pyright: ignore[reportUnknownVariableType]

from scrambler import KeyEntry

METRICS = ("accuracy", "acceptability", "readability", "weighted_tqa")


@dataclass
class _CLIArgs:
    evaluated_csv: str
    key_json: str
    out: str


class Metrics(TypedDict):
    accuracy: list[float]
    acceptability: list[float]
    readability: list[float]
    weighted_tqa: list[float]


class UnscrambledRow(TypedDict):
    text_id: int
    source_text: str
    baseline_translation: str
    ce_translation: str
    baseline_accuracy: float
    baseline_acceptability: float
    baseline_readability: float
    baseline_weighted_tqa: float
    ce_accuracy: float
    ce_acceptability: float
    ce_readability: float
    ce_weighted_tqa: float


def calculate_tqa(acc: float, accp: float, read: float) -> float:
    return ((acc * 3) + (accp * 2) + (read * 1)) / 6


def save_stat_report(
    filename: str, metric_name: str, base_scores: list[float], ce_scores: list[float]
):
    base_mean = np.mean(base_scores)
    ce_mean = np.mean(ce_scores)

    if np.array_equal(base_scores, ce_scores):
        stat, p_val = 0.0, 1.0
    else:
        try:
            stat, p_val = wilcoxon(base_scores, ce_scores)
        except ValueError:
            stat, p_val = np.nan, np.nan

    with open(filename, "w", encoding="utf-8") as f:
        _ = f.write(f"=== Wilcoxon Signed-Rank Test: {metric_name} ===\n")
        _ = f.write(f"Baseline Mean : {base_mean:.4f}\n")
        _ = f.write(f"CE Mean       : {ce_mean:.4f}\n")
        _ = f.write(f"Statistic     : {stat}\n")
        _ = f.write(f"p-value       : {p_val}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decode evaluation matrix and run Nababan TQA statistical analysis."
    )
    _ = parser.add_argument("evaluated_csv", help="Path to the scored blind_test.csv")
    _ = parser.add_argument("key_json", help="Path to the blind_key.json mapping key")
    _ = parser.add_argument(
        "-o",
        "--out",
        default="unscrambled_results.csv",
        help="Path to save the unscrambled dataset",
    )
    args = parser.parse_args(namespace=_CLIArgs)

    with open(args.key_json, "r", encoding="utf-8") as f:
        key_mapping = cast("dict[str, KeyEntry]", json.load(f))

    data_store: dict[str, Metrics] = {
        "baseline": Metrics(
            accuracy=[],
            acceptability=[],
            readability=[],
            weighted_tqa=[],
        ),
        "ce": Metrics(
            accuracy=[],
            acceptability=[],
            readability=[],
            weighted_tqa=[],
        ),
    }

    unscrambled_rows: list[UnscrambledRow] = []

    with open(args.evaluated_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pair_id"]
            if pid not in key_mapping:
                continue

            mapping = key_mapping[pid]

            try:
                acc_A = float(row["accuracy_A"])
                accp_A = float(row["acceptability_A"])
                read_A = float(row["readability_A"])

                acc_B = float(row["accuracy_B"])
                accp_B = float(row["acceptability_B"])
                read_B = float(row["readability_B"])
            except (ValueError, TypeError):
                print(
                    f"Warning: Missing or malformed data at pair_id {pid}. Skipping row.",
                    file=sys.stderr,
                )
                continue

            tqa_A = calculate_tqa(acc_A, accp_A, read_A)
            tqa_B = calculate_tqa(acc_B, accp_B, read_B)

            if mapping["translation_A"] == "baseline":
                data_store["baseline"]["accuracy"].append(acc_A)
                data_store["baseline"]["acceptability"].append(accp_A)
                data_store["baseline"]["readability"].append(read_A)
                data_store["baseline"]["weighted_tqa"].append(tqa_A)

                data_store["ce"]["accuracy"].append(acc_B)
                data_store["ce"]["acceptability"].append(accp_B)
                data_store["ce"]["readability"].append(read_B)
                data_store["ce"]["weighted_tqa"].append(tqa_B)
            else:
                data_store["ce"]["accuracy"].append(acc_A)
                data_store["ce"]["acceptability"].append(accp_A)
                data_store["ce"]["readability"].append(read_A)
                data_store["ce"]["weighted_tqa"].append(tqa_A)

                data_store["baseline"]["accuracy"].append(acc_B)
                data_store["baseline"]["acceptability"].append(accp_B)
                data_store["baseline"]["readability"].append(read_B)
                data_store["baseline"]["weighted_tqa"].append(tqa_B)

            unscrambled_rows.append(
                UnscrambledRow(
                    text_id=mapping["text_id"],
                    source_text=row["source_text"],
                    baseline_translation=row["translation_B"],
                    ce_translation=row["translation_A"],
                    baseline_accuracy=data_store["baseline"]["accuracy"][-1],
                    baseline_acceptability=data_store["baseline"]["acceptability"][-1],
                    baseline_readability=data_store["baseline"]["readability"][-1],
                    baseline_weighted_tqa=data_store["baseline"]["weighted_tqa"][-1],
                    ce_accuracy=data_store["ce"]["accuracy"][-1],
                    ce_acceptability=data_store["ce"]["acceptability"][-1],
                    ce_readability=data_store["ce"]["readability"][-1],
                    ce_weighted_tqa=data_store["ce"]["weighted_tqa"][-1],
                )
            )

    unscrambled_headers = [
        "text_id",
        "source_text",
        "baseline_translation",
        "ce_translation",
        "baseline_accuracy",
        "baseline_acceptability",
        "baseline_readability",
        "baseline_weighted_tqa",
        "ce_accuracy",
        "ce_acceptability",
        "ce_readability",
        "ce_weighted_tqa",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unscrambled_headers)
        writer.writeheader()
        writer.writerows(unscrambled_rows)

    for m in METRICS:
        save_stat_report(
            f"wilcoxon_results_{m}.txt",
            m.upper(),
            data_store["baseline"][m],
            data_store["ce"][m],
        )

    print(
        "Analysis finished.\n"
        + f"- Unscrambled results saved to: {args.out}\n"
        + "- Statistical breakdown saved into matching 'wilcoxon_results_*.txt' assets."
    )


if __name__ == "__main__":
    main()
