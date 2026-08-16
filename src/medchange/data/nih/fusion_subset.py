from __future__ import annotations

import pandas as pd


def build_fusion_qwen_subset(
    dataframe: pd.DataFrame,
    num_pairs: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a deterministic temporal cohort for caching Qwen evidence.

    Sampling is approximately balanced across:
      - unchanged
      - single temporal change
      - multiple temporal changes

    Patient IDs are retained so M4.5.2 can perform a patient-aware
    train/test split after Qwen inference has been cached.
    """

    required = {
        "pair_id",
        "patient_id",
        "num_changed_findings",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Temporal dataframe missing required columns: "
            f"{sorted(missing)}"
        )

    if num_pairs < 3:
        raise ValueError(
            "num_pairs must be at least 3."
        )

    frame = dataframe.copy()

    frame["fusion_category"] = "multi_change"

    frame.loc[
        frame["num_changed_findings"] == 0,
        "fusion_category",
    ] = "unchanged"

    frame.loc[
        frame["num_changed_findings"] == 1,
        "fusion_category",
    ] = "single_change"

    categories = [
        "unchanged",
        "single_change",
        "multi_change",
    ]

    base = num_pairs // 3
    remainder = num_pairs % 3

    requested = {
        category: base
        for category in categories
    }

    for category in categories[:remainder]:
        requested[category] += 1

    sampled_frames = []

    for offset, category in enumerate(
        categories
    ):
        subset = frame[
            frame["fusion_category"]
            == category
        ]

        n = requested[category]

        if len(subset) < n:
            raise ValueError(
                f"Not enough '{category}' pairs: "
                f"requested={n}, available={len(subset)}."
            )

        sampled = subset.sample(
            n=n,
            random_state=seed + offset,
            replace=False,
        )

        sampled_frames.append(
            sampled
        )

    result = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    result = result.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    if len(result) != num_pairs:
        raise RuntimeError(
            "Fusion subset size mismatch."
        )

    if result["pair_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate pair IDs detected."
        )

    return result