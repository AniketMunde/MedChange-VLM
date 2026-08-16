import pandas as pd
import pytest

from medchange.data.nih.fusion_subset import (
    build_fusion_qwen_subset,
)


def _make_dataframe():
    rows = []

    patient = 0

    for category, changed in [
        ("unchanged", 0),
        ("single", 1),
        ("multi", 2),
    ]:
        for index in range(100):
            rows.append(
                {
                    "pair_id": (
                        f"{category}_{index}"
                    ),
                    "patient_id": patient,
                    "num_changed_findings": changed,
                }
            )

            patient += 1

    return pd.DataFrame(rows)


def test_fusion_subset_size():
    result = build_fusion_qwen_subset(
        _make_dataframe(),
        num_pairs=200,
        seed=42,
    )

    assert len(result) == 200

    counts = (
        result["fusion_category"]
        .value_counts()
        .to_dict()
    )

    assert counts["unchanged"] == 67
    assert counts["single_change"] == 67
    assert counts["multi_change"] == 66


def test_fusion_subset_is_reproducible():
    dataframe = _make_dataframe()

    first = build_fusion_qwen_subset(
        dataframe,
        num_pairs=30,
        seed=42,
    )

    second = build_fusion_qwen_subset(
        dataframe,
        num_pairs=30,
        seed=42,
    )

    assert (
        first["pair_id"].tolist()
        == second["pair_id"].tolist()
    )


def test_fusion_subset_rejects_missing_columns():
    dataframe = pd.DataFrame(
        {
            "pair_id": ["a"],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_fusion_qwen_subset(
            dataframe
        )