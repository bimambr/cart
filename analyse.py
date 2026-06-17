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
from scipy.stats import wilcoxon, rankdata

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
    target_idiom: str
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
    base_arr = np.array(base_scores, dtype=np.float64)
    ce_arr = np.array(ce_scores, dtype=np.float64)
    base_median = cast(float, np.median(base_arr))
    ce_median = cast(float, np.median(ce_arr))
    base_mean = np.mean(base_arr)
    ce_mean = np.mean(ce_arr)
    diffs = ce_arr - base_arr
    nonzero = diffs[diffs != 0]
    n_total = len(diffs)
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_active = len(nonzero)
    n_zero = n_total - n_active
    median_diff = np.median(diffs)
    W_plus, W_minus, stat, p_val, rank_biserial = np.nan, np.nan, np.nan, 1.0, np.nan

    if nonzero.size > 0:
        try:
            res = wilcoxon(ce_arr, base_arr)
            stat, p_val = res.statistic, res.pvalue
            ranks = rankdata(np.abs(nonzero))
            W_plus = np.sum(ranks[nonzero > 0])
            W_minus = np.sum(ranks[nonzero < 0])
            rank_biserial = (W_plus - W_minus) / (W_plus + W_minus)
        except ValueError:
            pass

    with open(filename, "w", encoding="utf-8") as f:
        # fmt: off
        _ = f.write(f"=== Wilcoxon Signed-Rank Test Report: {metric_name} ===\n\n")

        _ = f.write("--- DESCRIPTIVE STATISTICS ---\n")
        _ = f.write(f"Baseline (Mean / Median)        : {base_mean:.4f} / {base_median:.4f}\n")
        _ = f.write(f"Context-Engineered (Mean / Med) : {ce_mean:.4f} / {ce_median:.4f}\n")
        _ = f.write(f"Median of Differences           : {median_diff:.4f}\n\n")

        _ = f.write("--- SAMPLE DISTRIBUTION DATA ---\n")
        _ = f.write(f"Total Paired Items (N)          : {n_total}\n")
        _ = f.write(f"Excluded Zero-Differences       : {n_zero}\n")
        _ = f.write(f"Effective Sample Size (n_active): {n_active}\n")
        _ = f.write(f"Positive Changes (CE > Base)    : {n_pos}\n")
        _ = f.write(f"Negative Changes (CE < Base)    : {n_neg}\n\n")

        _ = f.write("--- INFERENTIAL TEST METRICS ---\n")
        _ = f.write(f"W+ / W- / min(W+, W-)           : {W_plus} / {W_minus} / {stat} \n")
        _ = f.write(f"p-value                         : {p_val:.6f}\n")
        _ = f.write(f"Rank-Biserial Effect Size (r)   : {rank_biserial:.4f}\n")


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

            a_is_baseline = mapping["translation_A"] == "baseline"

            if a_is_baseline:
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
                    target_idiom=row["target_idiom"],
                    baseline_translation=row["translation_A"]
                    if a_is_baseline
                    else row["translation_B"],
                    ce_translation=row["translation_B"]
                    if a_is_baseline
                    else row["translation_A"],
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
        "target_idiom",
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
