from __future__ import annotations

import argparse
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

from medchange.evaluation import (
    BiomedCLIPEvaluationRunner,
    EVALUATION_FINDINGS,
    evaluate_predictions,
    load_evaluation_manifest,
    validate_image_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BiomedCLIP "
            "on a labeled CXR manifest."
        )
    )

    parser.add_argument(
        "manifest",
        type=str,
        help=(
            "CSV evaluation manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/"
            "biomedclip"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = (
        load_evaluation_manifest(
            args.manifest
        )
    )

    validate_image_paths(
        dataframe
    )

    runner = (
        BiomedCLIPEvaluationRunner()
    )

    predictions = runner.run(
        dataframe
    )

    metrics = (
        evaluate_predictions(
            dataframe=predictions,
            labels=(
                EVALUATION_FINDINGS
            ),
            threshold=(
                args.threshold
            ),
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

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "BiomedCLIP Evaluation"
    )
    print("=" * 90)

    print(
        metrics.to_string(
            index=False
        )
    )

    print("=" * 90)

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Metrics    : "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()