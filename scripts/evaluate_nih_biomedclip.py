from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.data.nih import (
    TARGET_FINDINGS,
    get_nih_split_names,
)
from medchange.evaluation.dataset_summary import (
    build_dataset_summary,
)
from medchange.evaluation.evaluator import (
    evaluate_predictions,
)
from medchange.evaluation.nih_runner import (
    NIHEvaluationRunner,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream NIH ChestXray14 from Hugging Face "
            "and benchmark BiomedCLIP."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--view",
        choices=[
            "ALL",
            "AP",
            "PA",
        ],
        default="ALL",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/"
            "nih_biomedclip"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    available_splits = (
        get_nih_split_names()
    )

    if (
        args.split
        not in available_splits
    ):
        raise ValueError(
            f"Unknown split '{args.split}'. "
            f"Available: {available_splits}"
        )

    print("=" * 80)
    print(
        "MedChange-VLM — NIH BiomedCLIP Benchmark"
    )
    print("=" * 80)

    print(
        f"Split          : {args.split}"
    )

    print(
        f"Max samples    : {args.max_samples}"
    )

    print(
        f"View           : {args.view}"
    )

    print(
        f"Shuffle        : {not args.no_shuffle}"
    )

    print(
        f"Seed           : {args.seed}"
    )

    print("=" * 80)

    runner = (
        NIHEvaluationRunner()
    )

    predictions = runner.run(
        split=args.split,

        max_samples=(
            args.max_samples
        ),

        view=args.view,

        shuffle=(
            not args.no_shuffle
        ),

        seed=args.seed,

        shuffle_buffer=(
            args.shuffle_buffer
        ),
    )

    metrics = (
        evaluate_predictions(
            dataframe=predictions,
            labels=TARGET_FINDINGS,
            threshold=args.threshold,
        )
    )

    summary = (
        build_dataset_summary(
            dataframe=predictions,
            findings=TARGET_FINDINGS,
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    metrics_path = (
        output_dir
        / "metrics.csv"
    )

    summary_path = (
        output_dir
        / "dataset_summary.json"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "Evaluation results"
    )
    print("=" * 100)

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        f"Predictions : {predictions_path}"
    )

    print(
        f"Metrics     : {metrics_path}"
    )

    print(
        f"Summary     : {summary_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()