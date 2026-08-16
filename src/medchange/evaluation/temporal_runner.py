from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)
from medchange.data.nih.temporal_split import (
    patient_aware_temporal_split,
)
from medchange.models.temporal.classifier import (
    TemporalLogisticClassifier,
)
from medchange.models.temporal.embedding_cache import (
    EmbeddingCache,
)
from medchange.models.temporal.features import (
    build_current_only_vector,
    build_longitudinal_vector,
    build_temporal_embedding_features,
)
from medchange.models.vision import (
    BiomedCLIP,
)


def _get_embedding(
    image_index: str,
    resolver: NIHImageResolver,
    cache: EmbeddingCache,
    model: BiomedCLIP,
):
    if cache.contains(
        image_index
    ):
        return cache.load(
            image_index
        )

    path = resolver.resolve(
        image_index
    )

    embedding = (
        model.encode_image(
            path
        )
    )

    cache.save(
        image_index,
        embedding,
    )

    return embedding


def build_feature_dataframe(
    dataframe: pd.DataFrame,
    resolver: NIHImageResolver,
    cache: EmbeddingCache,
    model: BiomedCLIP,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Build current-only and longitudinal feature matrices.
    """

    current_vectors = []
    longitudinal_vectors = []

    total = len(
        dataframe
    )

    for index, row in (
        dataframe.iterrows()
    ):
        prior_embedding = (
            _get_embedding(
                str(
                    row[
                        "prior_image_index"
                    ]
                ),
                resolver,
                cache,
                model,
            )
        )

        current_embedding = (
            _get_embedding(
                str(
                    row[
                        "current_image_index"
                    ]
                ),
                resolver,
                cache,
                model,
            )
        )

        features = (
            build_temporal_embedding_features(
                prior_embedding,
                current_embedding,
            )
        )

        current_vectors.append(
            build_current_only_vector(
                features
            )
        )

        longitudinal_vectors.append(
            build_longitudinal_vector(
                features
            )
        )

        processed = len(
            current_vectors
        )

        if (
            processed == 1
            or processed % 50 == 0
            or processed == total
        ):
            print(
                f"[{processed}/{total}] "
                "temporal embeddings prepared"
            )

    return (
        np.stack(
            current_vectors
        ),
        np.stack(
            longitudinal_vectors
        ),
    )


def run_temporal_experiment(
    dataframe: pd.DataFrame,
    dataset_root: str | Path,
    output_dir: str | Path,
    findings: list[str],
    seed: int = 42,
) -> dict:

    resolver = (
        NIHImageResolver(
            dataset_root
        )
    )

    resolver.build_index()

    print(
        f"Indexed {resolver.num_images:,} NIH images."
    )

    model = (
        BiomedCLIP()
    )

    cache = (
        EmbeddingCache(
            "data/nih/"
            "embedding_cache"
        )
    )

    split = patient_aware_temporal_split(
        dataframe,
        seed=seed,
    )

    combined = pd.concat(
        [
            split.train,
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )

    print()
    print("Preparing shared temporal feature matrix...")

    current_x, longitudinal_x = (
        build_feature_dataframe(
            combined,
            resolver,
            cache,
            model,
        )
    )

    n_train = len(split.train)
    n_validation = len(split.validation)

    train_end = n_train
    validation_end = (
            n_train + n_validation
    )

    results = {}

    for finding in findings:
        print()
        print("=" * 80)
        print(
            f"Temporal experiment: {finding}"
        )
        print("=" * 80)

        target_column = (
            f"{finding}_temporal"
        )

        y = (
            combined[target_column]
            .astype(str)
            .to_numpy()
        )

        y_train = y[:train_end]

        y_validation = y[
            train_end:validation_end
        ]

        y_test = y[
            validation_end:
        ]

        current_train = current_x[
            :train_end
        ]

        current_validation = current_x[
            train_end:validation_end
        ]

        current_test = current_x[
            validation_end:
        ]

        longitudinal_train = longitudinal_x[
            :train_end
        ]

        longitudinal_validation = longitudinal_x[
            train_end:validation_end
        ]

        longitudinal_test = longitudinal_x[
            validation_end:
        ]

        print(
            "Train classes:",
            pd.Series(
                y_train
            ).value_counts().to_dict(),
        )

        print(
            "Validation classes:",
            pd.Series(
                y_validation
            ).value_counts().to_dict(),
        )

        print(
            "Test classes:",
            pd.Series(
                y_test
            ).value_counts().to_dict(),
        )

        current_classifier = (
            TemporalLogisticClassifier(
                random_state=seed
            )
        )

        current_classifier.fit(
            current_train,
            y_train,
        )

        current_result = (
            current_classifier.evaluate(
                current_test,
                y_test,
            )
        )

        longitudinal_classifier = (
            TemporalLogisticClassifier(
                random_state=seed
            )
        )

        longitudinal_classifier.fit(
            longitudinal_train,
            y_train,
        )

        longitudinal_result = (
            longitudinal_classifier.evaluate(
                longitudinal_test,
                y_test,
            )
        )

        results[finding] = {
            "current_only": {
                "macro_f1":
                    current_result.macro_f1,

                "balanced_accuracy":
                    current_result.balanced_accuracy,

                "classification_report":
                    current_result.classification_report,

                "confusion_matrix":
                    current_result.confusion_matrix,
            },

            "longitudinal": {
                "macro_f1":
                    longitudinal_result.macro_f1,

                "balanced_accuracy":
                    longitudinal_result.balanced_accuracy,

                "classification_report":
                    longitudinal_result.classification_report,

                "confusion_matrix":
                    longitudinal_result.confusion_matrix,
            },

            "delta_macro_f1": (
                    longitudinal_result.macro_f1
                    - current_result.macro_f1
            ),
        }

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "temporal_results.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    return results