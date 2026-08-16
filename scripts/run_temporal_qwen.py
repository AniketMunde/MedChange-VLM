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

if str(
    SRC_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC_DIR
        ),
    )


from medchange.inference.temporal_pipeline import (
    TemporalQwenPipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare prior/current chest X-rays "
            "with Qwen2.5-VL."
        )
    )

    parser.add_argument(
        "--prior",
        required=True,
    )

    parser.add_argument(
        "--current",
        required=True,
    )

    parser.add_argument(
        "--pair-id",
        required=True,
    )

    parser.add_argument(
        "--prior-study-id",
        default="prior",
    )

    parser.add_argument(
        "--current-study-id",
        default="current",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = (
        TemporalQwenPipeline()
    )

    (
        prediction,
        metrics,
    ) = pipeline.analyze_pair(
        prior_image_path=(
            args.prior
        ),

        current_image_path=(
            args.current
        ),

        pair_id=(
            args.pair_id
        ),

        prior_study_id=(
            args.prior_study_id
        ),

        current_study_id=(
            args.current_study_id
        ),
    )

    print()
    print("=" * 90)
    print(
        "MedChange-VLM — "
        "Qwen Longitudinal Comparison"
    )
    print("=" * 90)

    print(
        json.dumps(
            prediction.model_dump(),
            indent=2,
        )
    )

    print()
    print(
        "Inference metrics"
    )

    print("-" * 90)

    print(
        f"Elapsed time : "
        f"{metrics.elapsed_seconds:.2f}s"
    )

    if (
        metrics.gpu_peak_allocated_gb
        is not None
    ):
        print(
            f"GPU peak     : "
            f"{metrics.gpu_peak_allocated_gb:.2f} GB"
        )


if __name__ == "__main__":
    main()