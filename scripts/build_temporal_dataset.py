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


from medchange.data.nih.temporal_dataset import (
    build_temporal_dataframe,
    load_pair_manifest,
)
from medchange.data.nih.temporal_summary import (
    summarize_temporal_dataset,
)
from medchange.data.nih.temporal_sampling import (
    sample_temporal_evaluation_subset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build MedChange temporal targets "
            "from NIH longitudinal pairs."
        )
    )

    parser.add_argument(
        "--pairs",
        type=str,
        default=(
            "data/nih/"
            "longitudinal_pairs_same_view.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/nih/"
            "temporal_pairs_same_view.csv"
        ),
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=(
            "experiments/"
            "nih_temporal/"
            "temporal_summary.json"
        ),
    )
    parser.add_argument(
        "--eval-subset",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval-output",
        type=str,
        default=(
            "data/nih/"
            "temporal_eval_subset.csv"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pair_manifest = (
        load_pair_manifest(
            args.pairs
        )
    )

    temporal = (
        build_temporal_dataframe(
            pair_manifest
        )
    )
    evaluation_subset = (
        sample_temporal_evaluation_subset(
            dataframe=temporal,
            total_pairs=(
                args.eval_subset
            ),
            changed_fraction=0.5,
            seed=42,
        )
    )

    evaluation_output = Path(
        args.eval_output
    )

    evaluation_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_subset.to_csv(
        evaluation_output,
        index=False,
    )

    summary = (
        summarize_temporal_dataset(
            temporal
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal.to_csv(
        output_path,
        index=False,
    )

    summary_path = Path(
        args.summary
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "MedChange-VLM Temporal Dataset"
    )
    print("=" * 80)

    print(
        f"Pairs               : "
        f"{summary['num_pairs']:,}"
    )

    print(
        f"Patients            : "
        f"{summary['num_patients']:,}"
    )
    print(
        f"Evaluation subset   : "
        f"{evaluation_output}"
    )

    print(
        f"Pairs with change   : "
        f"{summary['pairs_with_change']:,}"
    )

    print(
        f"Pairs without change: "
        f"{summary['pairs_without_change']:,}"
    )

    print(
        f"Mean changed finding: "
        f"{summary['mean_changed_findings']:.3f}"
    )

    print(
        f"Output              : "
        f"{output_path}"
    )

    print(
        f"Summary             : "
        f"{summary_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()