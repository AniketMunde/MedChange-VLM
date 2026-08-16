from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fusion_inputs(
    temporal_pairs_path: str | Path,
    qwen_pair_cache_path: str | Path,
    qwen_finding_cache_path: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Merge:
    - temporal ground truth / image IDs
    - successful Qwen pair cache
    - finding-level Qwen evidence
    """

    temporal = pd.read_csv(
        temporal_pairs_path
    )

    qwen_pairs = pd.read_csv(
        qwen_pair_cache_path
    )

    qwen_findings = pd.read_csv(
        qwen_finding_cache_path
    )

    required_temporal = {
        "pair_id",
        "patient_id",
        "prior_image_index",
        "current_image_index",
    }

    missing = (
        required_temporal
        - set(
            temporal.columns
        )
    )

    if missing:
        raise ValueError(
            "Temporal pair table missing columns: "
            f"{sorted(missing)}"
        )

    successful_ids = set(
        qwen_pairs[
            "pair_id"
        ]
        .astype(str)
    )

    temporal = temporal.copy()

    temporal[
        "pair_id"
    ] = temporal[
        "pair_id"
    ].astype(str)

    temporal = temporal[
        temporal[
            "pair_id"
        ].isin(
            successful_ids
        )
    ].copy()

    if temporal.empty:
        raise ValueError(
            "No temporal pairs match "
            "the Qwen cache."
        )

    duplicates = (
        qwen_findings
        .duplicated(
            [
                "pair_id",
                "finding",
            ]
        )
        .sum()
    )

    if duplicates:
        raise ValueError(
            "Duplicate Qwen finding evidence "
            f"rows detected: {duplicates}"
        )

    return (
        temporal,
        qwen_findings,
    )