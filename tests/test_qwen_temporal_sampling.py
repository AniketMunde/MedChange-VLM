import pandas as pd

from medchange.data.nih.qwen_temporal_sampling import (
    sample_qwen_temporal_benchmark,
)


def test_qwen_temporal_sampling():
    dataframe = pd.DataFrame(
        {
            "pair_id": [
                f"pair-{i}"
                for i in range(
                    30
                )
            ],

            "patient_id": [
                str(i)
                for i in range(
                    30
                )
            ],

            "has_temporal_change": (
                [False] * 10
                + [True] * 20
            ),

            "num_changed_findings": (
                [0] * 10
                + [1] * 10
                + [2] * 10
            ),
        }
    )

    result = (
        sample_qwen_temporal_benchmark(
            dataframe,
            unchanged_pairs=5,
            single_change_pairs=5,
            multi_change_pairs=5,
            seed=42,
        )
    )

    assert len(
        result
    ) == 15

    counts = (
        result[
            "benchmark_category"
        ]
        .value_counts()
        .to_dict()
    )

    assert (
        counts[
            "unchanged"
        ]
        == 5
    )

    assert (
        counts[
            "single_change"
        ]
        == 5
    )

    assert (
        counts[
            "multi_change"
        ]
        == 5
    )