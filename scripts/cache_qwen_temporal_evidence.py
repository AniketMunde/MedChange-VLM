from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from medchange.evaluation.qwen_evidence_cache import (
    QwenTemporalEvidenceCache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache Qwen temporal evidence "
            "for M4.5 fusion experiments."
        )
    )

    parser.add_argument(
        "--pairs",
        default=(
            "data/nih/"
            "fusion_qwen_subset_200.csv"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/"
            "temporal_fusion_m45/"
            "qwen_cache"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = pd.read_csv(
        args.pairs
    )

    runner = (
        QwenTemporalEvidenceCache(
            dataset_root=(
                args.dataset_root
            )
        )
    )

    metadata = runner.run(
        dataframe=dataframe,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print()
    print("=" * 90)
    print(
        "M4.5.1 CACHE SUMMARY"
    )
    print("=" * 90)

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("=" * 90)


if __name__ == "__main__":
    main()