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


from medchange.evaluation.temporal_aggregate import (
    build_final_temporal_summary,
    load_all_seed_summaries,
)


SEED_PATHS = {
    11: (
        "experiments/"
        "temporal_ablation/"
        "seed_11/"
        "temporal_ablation_summary.csv"
    ),

    21: (
        "experiments/"
        "temporal_ablation/"
        "seed_21/"
        "temporal_ablation_summary.csv"
    ),

    42: (
        "experiments/"
        "temporal_ablation/"
        "seed_42/"
        "temporal_ablation_summary.csv"
    ),

    84: (
        "experiments/"
        "temporal_ablation/"
        "seed_84/"
        "temporal_ablation_summary.csv"
    ),

    123: (
        "experiments/"
        "temporal_ablation/"
        "seed_123/"
        "temporal_ablation_summary.csv"
    ),
}


def main():
    combined = (
        load_all_seed_summaries(
            SEED_PATHS
        )
    )

    (
        aggregate,
        final_summary,
    ) = build_final_temporal_summary(
        combined
    )

    output_dir = Path(
        "experiments/"
        "temporal_ablation/"
        "aggregate"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_path = (
        output_dir
        / "all_seed_results.csv"
    )

    aggregate_path = (
        output_dir
        / "feature_aggregate.csv"
    )

    final_path = (
        output_dir
        / "final_temporal_summary.csv"
    )

    json_path = (
        output_dir
        / "final_temporal_summary.json"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    final_summary.to_csv(
        final_path,
        index=False,
    )

    records = (
        final_summary
        .to_dict(
            orient="records"
        )
    )

    json_path.write_text(
        json.dumps(
            records,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 125)
    print(
        "M4.3.2 — MULTI-SEED TEMPORAL ABLATION"
    )
    print("=" * 125)

    print(
        f"{'Finding':<22}"
        f"{'Current F1':>14}"
        f"{'Best temporal':>24}"
        f"{'Temporal F1':>16}"
        f"{'Delta':>12}"
        f"{'Wins':>10}"
    )

    print("-" * 125)

    for _, row in (
        final_summary.iterrows()
    ):
        current = (
            f"{row['current_macro_f1_mean']:.3f}"
            f" ± "
            f"{row['current_macro_f1_std']:.3f}"
        )

        temporal = (
            f"{row['best_temporal_macro_f1_mean']:.3f}"
            f" ± "
            f"{row['best_temporal_macro_f1_std']:.3f}"
        )

        wins = (
            f"{int(row['temporal_wins_vs_current'])}"
            f"/"
            f"{int(row['num_seeds'])}"
        )

        print(
            f"{row['finding']:<22}"
            f"{current:>14}"
            f"{row['best_temporal_feature']:>24}"
            f"{temporal:>16}"
            f"{row['delta_macro_f1']:>+12.4f}"
            f"{wins:>10}"
        )

    print("=" * 125)

    print()
    print(
        f"All seeds : {combined_path}"
    )

    print(
        f"Aggregate : {aggregate_path}"
    )

    print(
        f"Final CSV : {final_path}"
    )

    print(
        f"Final JSON: {json_path}"
    )


if __name__ == "__main__":
    main()