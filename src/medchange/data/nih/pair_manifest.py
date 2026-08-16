from __future__ import annotations

import pandas as pd

from medchange.data.nih.change_labels import (
    derive_temporal_label_change,
)
from medchange.data.nih.longitudinal import (
    NIHLongitudinalPair,
)


def build_pair_manifest(
    pairs: list[
        NIHLongitudinalPair
    ],
) -> pd.DataFrame:

    rows: list[dict] = []

    for pair in pairs:
        change = (
            derive_temporal_label_change(
                prior_labels=(
                    pair.prior_labels
                ),
                current_labels=(
                    pair.current_labels
                ),
            )
        )

        rows.append(
            {
                "pair_id": (
                    pair.pair_id
                ),

                "patient_id": (
                    pair.patient_id
                ),

                "prior_image_index": (
                    pair.prior_image_index
                ),

                "current_image_index": (
                    pair.current_image_index
                ),

                "prior_follow_up": (
                    pair.prior_follow_up
                ),

                "current_follow_up": (
                    pair.current_follow_up
                ),

                "follow_up_delta": (
                    pair.follow_up_delta
                ),

                "prior_view": (
                    pair.prior_view
                ),

                "current_view": (
                    pair.current_view
                ),

                "same_view": (
                    pair.same_view
                ),

                "prior_labels": "|".join(
                    pair.prior_labels
                ),

                "current_labels": "|".join(
                    pair.current_labels
                ),

                "new_findings": "|".join(
                    change.new
                ),

                "resolved_findings": "|".join(
                    change.resolved
                ),

                "persistent_findings": "|".join(
                    change.persistent
                ),

                "has_label_change": (
                    change.has_change
                ),
            }
        )

    return pd.DataFrame(
        rows
    )