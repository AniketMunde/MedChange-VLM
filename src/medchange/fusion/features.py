from __future__ import annotations

import numpy as np
import pandas as pd

from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)
from medchange.fusion.config import (
    BEST_BIOMEDCLIP_FEATURES,
)
from medchange.fusion.encoding import (
    build_qwen_feature_vector,
)
from medchange.models.temporal.ablation_features import (
    build_ablation_vector,
)
from medchange.models.temporal.embedding_cache import (
    EmbeddingCache,
)
from medchange.models.temporal.features import (
    build_temporal_embedding_features,
)


def build_pair_biomedclip_features(
    dataframe: pd.DataFrame,
    resolver: NIHImageResolver,
    cache: EmbeddingCache,
) -> dict[
    str,
    dict[
        str,
        np.ndarray,
    ],
]:
    """
    Build one BiomedCLIP feature vector per pair,
    for every pathology-specific frozen feature set.

    No Qwen inference and no BiomedCLIP inference
    should occur if embeddings are already cached.
    """

    result = {}

    total = len(
        dataframe
    )

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        pair_id = str(
            row[
                "pair_id"
            ]
        )

        prior_index = str(
            row[
                "prior_image_index"
            ]
        )

        current_index = str(
            row[
                "current_image_index"
            ]
        )

        if not cache.contains(
            prior_index
        ):
            raise FileNotFoundError(
                "Missing prior BiomedCLIP embedding: "
                f"{prior_index}"
            )

        if not cache.contains(
            current_index
        ):
            raise FileNotFoundError(
                "Missing current BiomedCLIP embedding: "
                f"{current_index}"
            )

        prior_embedding = (
            cache.load(
                prior_index
            )
        )

        current_embedding = (
            cache.load(
                current_index
            )
        )

        temporal_features = (
            build_temporal_embedding_features(
                prior_embedding,
                current_embedding,
            )
        )

        finding_vectors = {}

        for (
            finding,
            feature_set,
        ) in (
            BEST_BIOMEDCLIP_FEATURES
            .items()
        ):
            finding_vectors[
                finding
            ] = (
                build_ablation_vector(
                    temporal_features,
                    feature_set,
                )
            )

        result[
            pair_id
        ] = finding_vectors

        if (
            position == 1
            or position % 25 == 0
            or position == total
        ):
            print(
                f"[{position}/{total}] "
                "BiomedCLIP fusion features loaded"
            )

    return result


def build_qwen_lookup(
    qwen_findings: pd.DataFrame,
) -> dict[
    tuple[
        str,
        str,
    ],
    np.ndarray,
]:
    lookup = {}

    for _, row in (
        qwen_findings
        .iterrows()
    ):
        key = (
            str(
                row[
                    "pair_id"
                ]
            ),
            str(
                row[
                    "finding"
                ]
            ),
        )

        lookup[
            key
        ] = build_qwen_feature_vector(
            state=str(
                row[
                    "qwen_state"
                ]
            ),

            confidence=float(
                row[
                    "qwen_confidence"
                ]
            ),
        )

    return lookup