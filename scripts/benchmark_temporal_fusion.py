from __future__ import annotations

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


from medchange.evaluation.fusion_benchmark import (
    aggregate_fusion_results,
    run_fusion_benchmark,
    aggregate_models,
)


SEEDS = [
    11,
    21,
    42,
    84,
    123,
]


def main():

    output_dir = Path(
        "experiments/"
        "temporal_fusion_m45/"
        "benchmark"
    )

    results = (
        run_fusion_benchmark(
            temporal_pairs_path=(
                "data/nih/"
                "fusion_qwen_subset_200.csv"
            ),

            qwen_pair_cache_path=(
                "experiments/"
                "temporal_fusion_m45/"
                "qwen_cache/"
                "qwen_pair_cache.csv"
            ),

            qwen_finding_cache_path=(
                "experiments/"
                "temporal_fusion_m45/"
                "qwen_cache/"
                "qwen_finding_evidence.csv"
            ),

            embedding_cache_dir=(
                "data/nih/"
                "embedding_cache"
            ),

            output_dir=(
                output_dir
            ),

            seeds=SEEDS,
        )
    )

    aggregate = (
        aggregate_fusion_results(
            results
        )
    )
    model_summary = aggregate_models(
        results
    )

    model_summary.to_csv(
        output_dir
        / "model_summary.csv",
        index=False,
    )

    aggregate.to_csv(
        output_dir
        / "fusion_aggregate.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print(
        "M4.5.2 — BIOMEDCLIP + QWEN FUSION"
    )
    print("=" * 120)

    for finding in sorted(
        aggregate[
            "finding"
        ].unique()
    ):

        print()
        print(
            finding.upper()
        )

        subset = aggregate[
            aggregate[
                "finding"
            ]
            == finding
        ]
        print()
        print("OVERALL MODEL SUMMARY")
        print("-" * 80)

        for _, row in (
            subset.iterrows()
        ):
            print(
                f"{row['model']:<12}"
                f"Macro F1 = "
                f"{row['macro_f1_mean']:.3f}"
                f" ± "
                f"{row['macro_f1_std']:.3f}"
                f" | "
                f"Bal Acc = "
                f"{row['balanced_accuracy_mean']:.3f}"
            )

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()