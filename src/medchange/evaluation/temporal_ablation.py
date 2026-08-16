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
from medchange.models.temporal.ablation_features import (
    ABLATION_FEATURE_SETS,
    build_ablation_vector,
)
from medchange.models.temporal.classifier import (
    TemporalLogisticClassifier,
)
from medchange.models.temporal.embedding_cache import (
    EmbeddingCache,
)
from medchange.models.temporal.features import (
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

    embedding = model.encode_image(
        path
    )

    cache.save(
        image_index,
        embedding,
    )

    return embedding


def build_ablation_matrices(
    dataframe: pd.DataFrame,
    resolver: NIHImageResolver,
    cache: EmbeddingCache,
    model: BiomedCLIP,
) -> dict[str, np.ndarray]:
    """
    Build all feature matrices once.

    BiomedCLIP embeddings are reused from cache.
    """

    matrices: dict[
        str,
        list[np.ndarray],
    ] = {
        feature_set: []
        for feature_set
        in ABLATION_FEATURE_SETS
    }

    total = len(
        dataframe
    )

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
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

        for feature_set in (
            ABLATION_FEATURE_SETS
        ):
            vector = (
                build_ablation_vector(
                    features,
                    feature_set,
                )
            )

            matrices[
                feature_set
            ].append(
                vector
            )

        if (
            position == 1
            or position % 50 == 0
            or position == total
        ):
            print(
                f"[{position}/{total}] "
                "ablation features prepared"
            )

    return {
        feature_set: np.stack(
            vectors
        )
        for feature_set, vectors
        in matrices.items()
    }


def run_temporal_ablation(
    dataframe: pd.DataFrame,
    dataset_root: str | Path,
    findings: list[str],
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    """
    Compare temporal feature representations
    using identical patient-aware splits.
    """

    resolver = NIHImageResolver(
        dataset_root
    )

    resolver.build_index()

    print(
        f"Indexed {resolver.num_images:,} NIH images."
    )

    model = BiomedCLIP()

    cache = EmbeddingCache(
        "data/nih/embedding_cache"
    )

    split = (
        patient_aware_temporal_split(
            dataframe,
            seed=seed,
        )
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
    print(
        "Preparing shared ablation matrices..."
    )

    feature_matrices = (
        build_ablation_matrices(
            dataframe=combined,
            resolver=resolver,
            cache=cache,
            model=model,
        )
    )

    n_train = len(
        split.train
    )

    n_validation = len(
        split.validation
    )

    train_end = n_train

    validation_end = (
        n_train
        + n_validation
    )

    results: dict = {}

    summary_rows = []

    for finding in findings:
        print()
        print("=" * 90)
        print(
            f"Temporal ablation: {finding}"
        )
        print("=" * 90)

        target_column = (
            f"{finding}_temporal"
        )

        y = (
            combined[
                target_column
            ]
            .astype(str)
            .to_numpy()
        )

        y_train = y[
            :train_end
        ]

        y_test = y[
            validation_end:
        ]

        print(
            "Train classes:",
            pd.Series(
                y_train
            ).value_counts().to_dict(),
        )

        print(
            "Test classes:",
            pd.Series(
                y_test
            ).value_counts().to_dict(),
        )

        finding_results = {}

        for feature_set in (
            ABLATION_FEATURE_SETS
        ):
            matrix = (
                feature_matrices[
                    feature_set
                ]
            )

            x_train = matrix[
                :train_end
            ]

            x_test = matrix[
                validation_end:
            ]

            classifier = (
                TemporalLogisticClassifier(
                    random_state=seed
                )
            )

            classifier.fit(
                x_train,
                y_train,
            )

            result = (
                classifier.evaluate(
                    x_test,
                    y_test,
                )
            )

            finding_results[
                feature_set
            ] = {
                "macro_f1": (
                    result.macro_f1
                ),

                "balanced_accuracy": (
                    result
                    .balanced_accuracy
                ),

                "classification_report": (
                    result
                    .classification_report
                ),

                "confusion_matrix": (
                    result
                    .confusion_matrix
                ),
            }

            summary_rows.append(
                {
                    "finding": finding,
                    "feature_set": (
                        feature_set
                    ),
                    "macro_f1": (
                        result.macro_f1
                    ),
                    "balanced_accuracy": (
                        result
                        .balanced_accuracy
                    ),
                }
            )

            print(
                f"{feature_set:<24}"
                f"Macro F1="
                f"{result.macro_f1:.4f}  "
                f"Balanced Acc="
                f"{result.balanced_accuracy:.4f}"
            )

        best_feature = max(
            finding_results,
            key=lambda name: (
                finding_results[
                    name
                ][
                    "macro_f1"
                ]
            ),
        )

        finding_results[
            "best_feature_set"
        ] = best_feature

        finding_results[
            "best_macro_f1"
        ] = (
            finding_results[
                best_feature
            ][
                "macro_f1"
            ]
        )

        results[
            finding
        ] = finding_results

        print()
        print(
            "Best feature set:",
            best_feature,
        )

        print(
            "Best Macro F1  :",
            f"{finding_results['best_macro_f1']:.4f}",
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        output_dir
        / "temporal_ablation_results.json"
    )

    summary_path = (
        output_dir
        / "temporal_ablation_summary.csv"
    )

    detailed_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return results