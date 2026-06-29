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
from typing import Literal, TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
import scikit_posthocs as sp  # pyright: ignore[reportMissingTypeStubs]
from scipy.stats import friedmanchisquare, wilcoxon

from scrambler import KeyEntry

TREATMENTS = ("T1", "T2", "T3", "T4", "T5")
METRICS = ("accuracy", "acceptability", "readability", "weighted_tqa")
METRIC: TypeAlias = Literal["accuracy", "acceptability", "readability", "weighted_tqa"]


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
    T1_translation: str
    T2_translation: str
    T3_translation: str
    T4_translation: str
    T5_translation: str
    T1_accuracy: float
    T1_acceptability: float
    T1_readability: float
    T1_weighted_tqa: float
    T2_accuracy: float
    T2_acceptability: float
    T2_readability: float
    T2_weighted_tqa: float
    T3_accuracy: float
    T3_acceptability: float
    T3_readability: float
    T3_weighted_tqa: float
    T4_accuracy: float
    T4_acceptability: float
    T4_readability: float
    T4_weighted_tqa: float
    T5_accuracy: float
    T5_acceptability: float
    T5_readability: float
    T5_weighted_tqa: float


def calculate_tqa(acc: float, accp: float, read: float) -> float:
    return ((acc * 3) + (accp * 2) + (read * 1)) / 6


def run_factorial_wilcoxon(metric_name: METRIC, stores: dict[str, Metrics]):
    t2 = np.array(stores["T2"][metric_name])
    t3 = np.array(stores["T3"][metric_name])
    t4 = np.array(stores["T4"][metric_name])
    t5 = np.array(stores["T5"][metric_name])

    _, p_rag = wilcoxon(t5 + t3, t4 + t2)
    _, p_refine = wilcoxon(t5 + t4, t3 + t2)
    _, p_interaction = wilcoxon(t5 - t4, t3 - t2)

    with open(f"factorial_wilcoxon_{metric_name}.txt", "w", encoding="utf-8") as f:
        # fmt: off
        _ = f.write(f"=== Non-Parametric Factorial Analysis: {metric_name.upper()} ===\n\n")
        _ = f.write(f"Main Effect (RAG) p-value    : {p_rag:.6f}\n")
        _ = f.write(f"Main Effect (Refine) p-value : {p_refine:.6f}\n")
        _ = f.write(f"Interaction Effect p-value   : {p_interaction:.6f}\n")


def run_anova(
    metric_name: METRIC,
    stores: dict[str, Metrics],
):
    t1 = np.array(stores["T1"][metric_name])
    t2 = np.array(stores["T2"][metric_name])
    t3 = np.array(stores["T3"][metric_name])
    t4 = np.array(stores["T4"][metric_name])
    t5 = np.array(stores["T5"][metric_name])

    n_blocks = len(t1)

    stat, p_omnibus = friedmanchisquare(t1, t2, t3, t4, t5)
    df = pd.DataFrame(
        {
            "score": np.concatenate([t1, t2, t3, t4, t5]),
            "treatment": ["T1_Base"] * n_blocks
            + ["T2_-Ref-RAG"] * n_blocks
            + ["T3_-Ref+RAG"] * n_blocks
            + ["T4_+Ref-RAG"] * n_blocks
            + ["T5_+Ref+RAG"] * n_blocks,
            "block": list(range(n_blocks)) * 5,
        }
    )
    p_matrix = sp.posthoc_conover_friedman(  # pyright: ignore[reportUnknownMemberType]
        df,
        y_col="score",
        group_col="treatment",
        block_col="block",
        block_id_col="block",
        melted=True,
        p_adjust="holm",
    )

    with open(f"anova_results_{metric_name}.txt", "w", encoding="utf-8") as f:
        # fmt: off
        _ = f.write(f"=== Friedman ANOVA & Post-Hoc Report: {metric_name.upper()} ===\n\n")
        _ = f.write(f"Sample Size (N Blocks) : {n_blocks}\n")
        _ = f.write(f"Friedman Chi-Square    : {stat:.4f}\n")
        _ = f.write(f"Omnibus p-value        : {p_omnibus:.6f}\n\n")

        _ = f.write("--- DESCRIPTIVE STATISTICS (Means) ---\n")
        _ = f.write(f"T1 (Baseline Direct)   : {np.mean(t1):.4f}\n")
        _ = f.write(f"T2 (-Ref -RAG)         : {np.mean(t2):.4f}\n")
        _ = f.write(f"T3 (-Ref +RAG)         : {np.mean(t3):.4f}\n")
        _ = f.write(f"T4 (+Ref -RAG)         : {np.mean(t4):.4f}\n\n")
        _ = f.write(f"T5 (+Ref +RAG)         : {np.mean(t5):.4f}\n\n")

        _ = f.write("--- POST-HOC PAIRWISE CONOVER P-MATRICES (Holm-Adjusted) ---\n")
        _ = f.write(p_matrix.to_string())
        _ = f.write("\n")


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
        t: {
            "accuracy": [],
            "acceptability": [],
            "readability": [],
            "weighted_tqa": [],
        }
        for t in TREATMENTS
    }
    unscrambled_rows: list[UnscrambledRow] = []

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
                    "E": {
                        "acc": float(row["accuracy_E"]),
                        "accp": float(row["acceptability_E"]),
                        "read": float(row["readability_E"]),
                    },
                }
            except (ValueError, KeyError, TypeError):
                print(
                    f"Warning: Missing or malformed data at pair_id {pid}. Skipping row.",
                    file=sys.stderr,
                )
                continue

            tqas: dict[str, float] = {}
            for col, tx in mapping.items():
                tqas[tx] = tqa = calculate_tqa(
                    scores[col]["acc"],
                    scores[col]["accp"],
                    scores[col]["read"],
                )
                data_store[tx]["accuracy"].append(scores[col]["acc"])
                data_store[tx]["acceptability"].append(scores[col]["accp"])
                data_store[tx]["readability"].append(scores[col]["read"])
                data_store[tx]["weighted_tqa"].append(tqa)

            col_for = {tx: col for col, tx in mapping.items()}
            T1 = col_for["T1"]
            T2 = col_for["T2"]
            T3 = col_for["T3"]
            T4 = col_for["T4"]
            T5 = col_for["T5"]

            unscrambled_rows.append(
                UnscrambledRow(
                    text_id=key_mapping[pid]["text_id"],
                    source_text=row["source_text"],
                    target_idiom=row["target_idiom"],
                    T1_translation=row[f"translation_{T1}"],
                    T2_translation=row[f"translation_{T2}"],
                    T3_translation=row[f"translation_{T3}"],
                    T4_translation=row[f"translation_{T4}"],
                    T5_translation=row[f"translation_{T5}"],
                    T1_accuracy=scores[T1]["acc"],
                    T1_acceptability=scores[T1]["accp"],
                    T1_readability=scores[T1]["read"],
                    T1_weighted_tqa=tqas[T1],
                    T2_accuracy=scores[T2]["acc"],
                    T2_acceptability=scores[T2]["accp"],
                    T2_readability=scores[T2]["read"],
                    T2_weighted_tqa=tqas[T2],
                    T3_accuracy=scores[T3]["acc"],
                    T3_acceptability=scores[T3]["accp"],
                    T3_readability=scores[T3]["read"],
                    T3_weighted_tqa=tqas[T3],
                    T4_accuracy=scores[T4]["acc"],
                    T4_acceptability=scores[T4]["accp"],
                    T4_readability=scores[T4]["read"],
                    T4_weighted_tqa=tqas[T4],
                    T5_accuracy=scores[T5]["acc"],
                    T5_acceptability=scores[T5]["accp"],
                    T5_readability=scores[T5]["read"],
                    T5_weighted_tqa=tqas[T5],
                )
            )

    unscrambled_headers = [
        "text_id",
        "source_text",
        "target_idiom",
        "T1_translation",
        "T2_translation",
        "T3_translation",
        "T4_translation",
        "T5_translation",
        "T1_accuracy",
        "T1_acceptability",
        "T1_readability",
        "T1_weighted_tqa",
        "T2_accuracy",
        "T2_acceptability",
        "T2_readability",
        "T2_weighted_tqa",
        "T3_accuracy",
        "T3_acceptability",
        "T3_readability",
        "T3_weighted_tqa",
        "T4_accuracy",
        "T4_acceptability",
        "T4_readability",
        "T4_weighted_tqa",
        "T5_accuracy",
        "T5_acceptability",
        "T5_readability",
        "T5_weighted_tqa",
    ]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unscrambled_headers)
        writer.writeheader()
        writer.writerows(unscrambled_rows)

    for m in METRICS:
        run_anova(m, data_store)
        run_factorial_wilcoxon(m, data_store)

    print(
        "Analysis finished.\n"
        + f"- Unscrambled results saved to: {args.out}\n"
        + "- Statistical breakdown saved into matching 'anova_results_*.txt' and 'factorial_wilcoxon_*' assets."
    )


if __name__ == "__main__":
    main()
