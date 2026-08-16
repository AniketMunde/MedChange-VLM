from __future__ import annotations

import pandas as pd


def sample_qwen_temporal_benchmark(
    dataframe: pd.DataFrame,
    unchanged_pairs: int = 10,
    single_change_pairs: int = 10,
    multi_change_pairs: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Construct a deterministic Qwen temporal benchmark.

    Groups:
    - unchanged: zero NEW/RESOLVED target findings
    - single_change: exactly one changed target finding
    - multi_change: two or more changed target findings
    """

    required = {
        "pair_id",
        "patient_id",
        "has_temporal_change",
        "num_changed_findings",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Temporal dataset missing columns: "
            f"{sorted(missing)}"
        )

    unchanged = dataframe[
        dataframe[
            "num_changed_findings"
        ]
        == 0
    ]

    single_change = dataframe[
        dataframe[
            "num_changed_findings"
        ]
        == 1
    ]

    multi_change = dataframe[
        dataframe[
            "num_changed_findings"
        ]
        >= 2
    ]

    requests = [
        (
            "unchanged",
            unchanged,
            unchanged_pairs,
            seed,
        ),
        (
            "single_change",
            single_change,
            single_change_pairs,
            seed + 1,
        ),
        (
            "multi_change",
            multi_change,
            multi_change_pairs,
            seed + 2,
        ),
    ]

    samples = []

    for (
        category,
        subset,
        count,
        random_state,
    ) in requests:

        if len(
            subset
        ) < count:
            raise ValueError(
                f"Not enough {category} pairs. "
                f"Requested={count}, "
                f"available={len(subset)}"
            )

        sampled = subset.sample(
            n=count,
            random_state=random_state,
            replace=False,
        ).copy()

        sampled[
            "benchmark_category"
        ] = category

        samples.append(
            sampled
        )

    benchmark = pd.concat(
        samples,
        ignore_index=True,
    )

    benchmark = benchmark.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    return benchmark