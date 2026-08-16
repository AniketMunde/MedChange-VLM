from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


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
        str(
            SRC_DIR
        ),
    )


from medchange.evaluation.policy_tuning import (
    aggregate_policy_grid,
    run_policy_grid,
)


def main() -> None:
    input_path = Path(
        "experiments/"
        "medchange_m55_oof/"
        "oof_system_predictions.csv"
    )

    output_dir = Path(
        "experiments/"
        "medchange_m552"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        input_path
    )

    results = (
        run_policy_grid(
            dataframe
        )
    )

    aggregate = (
        aggregate_policy_grid(
            results
        )
    )

    results.to_csv(
        output_dir
        / "policy_seed_results.csv",
        index=False,
    )

    aggregate.to_csv(
        output_dir
        / "policy_aggregate.csv",
        index=False,
    )

    # -------------------------------------
    # Useful operating-point ranking
    # -------------------------------------

    candidates = aggregate[
        aggregate[
            "coverage_mean"
        ]
        >= 0.75
    ].copy()

    candidates = candidates[
        candidates[
            "new_recall_mean"
        ]
        >= 0.10
    ]

    candidates = candidates.sort_values(
        by=[
            "error_rate_mean",
            "selective_macro_f1_mean",
            "coverage_mean",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    candidates.to_csv(
        output_dir
        / "recommended_operating_points.csv",
        index=False,
    )

    print()
    print("=" * 125)
    print(
        "M5.5.2 — SELECTIVE-RISK "
        "POLICY TUNING"
    )
    print("=" * 125)

    print()
    print(
        f"{'Policy':<22}"
        f"{'Thr':<8}"
        f"{'Coverage':<12}"
        f"{'Accuracy':<12}"
        f"{'Macro F1':<12}"
        f"{'Error':<12}"
        f"{'NEW':<10}"
        f"{'PERSIST':<10}"
        f"{'RESOLVED'}"
    )

    print("-" * 125)

    for _, row in (
        aggregate.iterrows()
    ):
        print(
            f"{row['policy']:<22}"
            f"{row['threshold']:<8.2f}"
            f"{row['coverage_mean']:<12.3f}"
            f"{row['selective_accuracy_mean']:<12.3f}"
            f"{row['selective_macro_f1_mean']:<12.3f}"
            f"{row['error_rate_mean']:<12.3f}"
            f"{row['new_recall_mean']:<10.3f}"
            f"{row['persistent_recall_mean']:<10.3f}"
            f"{row['resolved_recall_mean']:.3f}"
        )

    print()
    print(
        "RECOMMENDED OPERATING POINTS"
    )

    print("-" * 125)

    if candidates.empty:
        print(
            "No configuration met the "
            "coverage/change-recall constraints."
        )

    else:
        print(
            candidates.head(
                10
            )[
                [
                    "policy",
                    "threshold",
                    "coverage_mean",
                    "selective_accuracy_mean",
                    "selective_macro_f1_mean",
                    "error_rate_mean",
                    "new_recall_mean",
                    "persistent_recall_mean",
                    "resolved_recall_mean",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Artifacts: "
        "experiments/medchange_m552"
    )

    print("=" * 125)


if __name__ == "__main__":
    main()