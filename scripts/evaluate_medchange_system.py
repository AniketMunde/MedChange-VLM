from __future__ import annotations

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


from medchange.evaluation.medchange_system_benchmark import (
    run_medchange_system_benchmark,
)


SEEDS = [
    11,
    21,
    42,
    84,
    123,
]


def _format_metric(
    row: dict,
    name: str,
) -> str:
    mean = row.get(
        f"{name}_mean"
    )

    std = row.get(
        f"{name}_std"
    )

    if mean is None:
        return "N/A"

    if std is None:
        return (
            f"{mean:.3f}"
        )

    return (
        f"{mean:.3f} "
        f"± {std:.3f}"
    )


def main() -> None:
    metrics = (
        run_medchange_system_benchmark(
            temporal_pairs_path=(
                "data/nih/"
                "fusion_qwen_subset_200.csv"
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
                "experiments/"
                "medchange_m55_oof"
            ),

            seeds=SEEDS,
        )
    )

    print()
    print("=" * 110)
    print(
        "M5.5.1 — PATIENT-AWARE "
        "OUT-OF-FOLD MEDCHANGE EVALUATION"
    )
    print("=" * 110)

    print(
        f"Source pairs        : "
        f"{metrics['num_source_pairs']}"
    )

    print(
        f"Seeds               : "
        f"{metrics['seeds']}"
    )

    print(
        f"Patient overlap     : "
        f"{metrics['patient_overlap_all_seeds']}"
    )

    print()

    for row in (
        metrics[
            "models"
        ]
    ):
        print(
            row[
                "model"
            ].upper()
        )

        print(
            "Coverage             : "
            + _format_metric(
                row,
                "coverage",
            )
        )

        print(
            "Abstention rate      : "
            + _format_metric(
                row,
                "abstention_rate",
            )
        )

        print(
            "Selective accuracy   : "
            + _format_metric(
                row,
                "selective_accuracy",
            )
        )

        print(
            "Selective Macro F1   : "
            + _format_metric(
                row,
                "selective_macro_f1",
            )
        )

        print(
            "Covered error rate   : "
            + _format_metric(
                row,
                "error_rate_on_covered",
            )
        )

        print(
            "Review rate          : "
            + _format_metric(
                row,
                "review_rate",
            )
        )

        print(
            "Absent recall        : "
            + _format_metric(
                row,
                "absent_recall",
            )
        )

        print(
            "New recall           : "
            + _format_metric(
                row,
                "new_recall",
            )
        )

        print(
            "Persistent recall    : "
            + _format_metric(
                row,
                "persistent_recall",
            )
        )

        print(
            "Resolved recall      : "
            + _format_metric(
                row,
                "resolved_recall",
            )
        )

        print()

    print(
        "PAIR-LEVEL MEDCHANGE METRICS"
    )

    print("-" * 110)

    pair = (
        metrics[
            "pair_metrics"
        ]
    )

    print(
        "Exact pair match      : "
        f"{pair['exact_pair_match_rate_mean']:.3f}"
        " ± "
        f"{pair['exact_pair_match_rate_std']:.3f}"
    )

    print(
        "Pair abstention rate  : "
        f"{pair['pair_abstention_rate_mean']:.3f}"
        " ± "
        f"{pair['pair_abstention_rate_std']:.3f}"
    )

    print(
        "Pair review rate      : "
        f"{pair['pair_review_rate_mean']:.3f}"
        " ± "
        f"{pair['pair_review_rate_std']:.3f}"
    )

    print(
        "Fully covered pairs   : "
        f"{pair['fully_covered_pair_rate_mean']:.3f}"
        " ± "
        f"{pair['fully_covered_pair_rate_std']:.3f}"
    )

    print(
        "Mean covered findings : "
        f"{pair['mean_covered_findings_per_pair_mean']:.3f}"
        " ± "
        f"{pair['mean_covered_findings_per_pair_std']:.3f}"
    )

    print()
    print(
        f"Conflict rate         : "
        f"{metrics['conflict_rate']:.3f}"
    )

    print(
        f"Uncertainty rate      : "
        f"{metrics['uncertainty_rate']:.3f}"
    )

    print()
    print(
        "Artifacts:"
    )

    print(
        "experiments/"
        "medchange_m55_oof"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()