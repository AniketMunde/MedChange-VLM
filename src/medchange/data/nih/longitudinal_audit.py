from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from medchange.data.nih.longitudinal import (
    NIHLongitudinalStudy,
    build_adjacent_pairs,
    group_studies_by_patient,
)


def audit_longitudinal_dataset(
    studies: list[
        NIHLongitudinalStudy
    ],
) -> dict[str, Any]:

    groups = (
        group_studies_by_patient(
            studies
        )
    )

    patient_study_counts = [
        len(
            patient_studies
        )
        for patient_studies
        in groups.values()
    ]

    repeated_patients = [
        count
        for count
        in patient_study_counts
        if count >= 2
    ]

    adjacent_pairs = (
        build_adjacent_pairs(
            studies,
            same_view_only=False,
        )
    )

    same_view_pairs = (
        build_adjacent_pairs(
            studies,
            same_view_only=True,
        )
    )

    follow_up_deltas = Counter(
        pair.follow_up_delta
        for pair
        in adjacent_pairs
    )

    view_transitions = Counter(
        (
            pair.prior_view,
            pair.current_view,
        )
        for pair
        in adjacent_pairs
    )

    return {
        "num_images": len(
            studies
        ),

        "num_patients": len(
            groups
        ),

        "num_repeated_patients": len(
            repeated_patients
        ),

        "fraction_repeated_patients": (
            len(
                repeated_patients
            )
            / len(
                groups
            )
            if groups
            else 0.0
        ),

        "max_studies_per_patient": (
            max(
                patient_study_counts
            )
            if patient_study_counts
            else 0
        ),

        "adjacent_pair_count": len(
            adjacent_pairs
        ),

        "same_view_pair_count": len(
            same_view_pairs
        ),

        "same_view_pair_fraction": (
            len(
                same_view_pairs
            )
            / len(
                adjacent_pairs
            )
            if adjacent_pairs
            else 0.0
        ),

        "follow_up_delta_counts": {
            str(key): int(value)
            for key, value
            in follow_up_deltas.items()
        },

        "view_transition_counts": {
            f"{prior}->{current}": int(
                count
            )
            for (
                prior,
                current,
            ), count
            in view_transitions.items()
        },
    }