from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medchange.data.nih.temporal_split import (
    patient_aware_temporal_split,
)
from medchange.evaluation.fusion_metrics import (
    evaluate_fusion_predictions,
)
from medchange.fusion.classifier import (
    FusionClassifier,
)
from medchange.fusion.config import (
    FUSION_FINDINGS,
)
from medchange.fusion.dataset import (
    load_fusion_inputs,
)
from medchange.fusion.features import (
    build_pair_biomedclip_features,
    build_qwen_lookup,
)
from medchange.models.temporal.embedding_cache import (
    EmbeddingCache,
)


QWEN_STATE_TO_LABEL = {
    "absent": "absent",
    "new": "new",
    "persistent": "persistent",
    "resolved": "resolved",
    "uncertain": "absent",
}


def _qwen_state_from_vector(
    vector: np.ndarray,
) -> str:

    state_index = int(
        np.argmax(
            vector[:5]
        )
    )

    states = [
        "absent",
        "new",
        "persistent",
        "resolved",
        "uncertain",
    ]

    return states[
        state_index
    ]


def run_fusion_benchmark(
    temporal_pairs_path: str | Path,
    qwen_pair_cache_path: str | Path,
    qwen_finding_cache_path: str | Path,
    embedding_cache_dir: str | Path,
    output_dir: str | Path,
    seeds: list[int],
) -> pd.DataFrame:

    (
        temporal,
        qwen_findings,
    ) = load_fusion_inputs(
        temporal_pairs_path=(
            temporal_pairs_path
        ),
        qwen_pair_cache_path=(
            qwen_pair_cache_path
        ),
        qwen_finding_cache_path=(
            qwen_finding_cache_path
        ),
    )

    cache = EmbeddingCache(
        embedding_cache_dir
    )

    pair_features = (
        build_pair_biomedclip_features(
            dataframe=temporal,
            resolver=None,
            cache=cache,
        )
    )

    qwen_lookup = (
        build_qwen_lookup(
            qwen_findings
        )
    )

    rows = []

    for seed in seeds:

        print()
        print("=" * 90)
        print(
            f"Fusion experiment seed={seed}"
        )
        print("=" * 90)

        split = (
            patient_aware_temporal_split(
                temporal,
                train_fraction=0.70,
                validation_fraction=0.15,
                seed=seed,
            )
        )

        train_ids = set(
            split.train[
                "pair_id"
            ].astype(str)
        )

        test_ids = set(
            split.test[
                "pair_id"
            ].astype(str)
        )

        train_patients = set(
            split.train[
                "patient_id"
            ].astype(str)
        )

        test_patients = set(
            split.test[
                "patient_id"
            ].astype(str)
        )

        overlap = (
            train_patients
            & test_patients
        )

        if overlap:
            raise RuntimeError(
                "Patient leakage detected "
                f"for seed {seed}."
            )

        for finding in (
            FUSION_FINDINGS
        ):

            target_column = (
                f"{finding}_temporal"
            )

            x_bio_train = []
            x_bio_test = []

            x_qwen_train = []
            x_qwen_test = []

            x_fusion_train = []
            x_fusion_test = []

            y_train = []
            y_test = []

            qwen_test_predictions = []

            for _, row in (
                temporal.iterrows()
            ):

                pair_id = str(
                    row[
                        "pair_id"
                    ]
                )

                if (
                    pair_id
                    not in train_ids
                    and pair_id
                    not in test_ids
                ):
                    continue

                bio_vector = (
                    pair_features[
                        pair_id
                    ][
                        finding
                    ]
                )

                qwen_key = (
                    pair_id,
                    finding,
                )

                if (
                    qwen_key
                    not in qwen_lookup
                ):
                    raise KeyError(
                        "Missing Qwen evidence: "
                        f"{qwen_key}"
                    )

                qwen_vector = (
                    qwen_lookup[
                        qwen_key
                    ]
                )

                fusion_vector = (
                    np.concatenate(
                        [
                            bio_vector,
                            qwen_vector,
                        ]
                    ).astype(
                        np.float32
                    )
                )

                target = str(
                    row[
                        target_column
                    ]
                )

                if pair_id in train_ids:

                    x_bio_train.append(
                        bio_vector
                    )

                    x_qwen_train.append(
                        qwen_vector
                    )

                    x_fusion_train.append(
                        fusion_vector
                    )

                    y_train.append(
                        target
                    )

                elif pair_id in test_ids:

                    x_bio_test.append(
                        bio_vector
                    )

                    x_qwen_test.append(
                        qwen_vector
                    )

                    x_fusion_test.append(
                        fusion_vector
                    )

                    y_test.append(
                        target
                    )

                    qwen_state = (
                        _qwen_state_from_vector(
                            qwen_vector
                        )
                    )

                    qwen_test_predictions.append(
                        QWEN_STATE_TO_LABEL[
                            qwen_state
                        ]
                    )

            y_train_array = np.asarray(
                y_train
            )

            y_test_array = np.asarray(
                y_test
            )

            # -------------------
            # BiomedCLIP-only
            # -------------------

            bio_classifier = (
                FusionClassifier(
                    seed=seed
                )
            )

            bio_classifier.fit(
                np.stack(
                    x_bio_train
                ),
                y_train_array,
            )

            bio_predictions = (
                bio_classifier.predict(
                    np.stack(
                        x_bio_test
                    )
                )
            )

            bio_metrics = (
                evaluate_fusion_predictions(
                    y_test_array,
                    bio_predictions,
                )
            )

            # -------------------
            # Qwen-only
            # -------------------

            qwen_metrics = (
                evaluate_fusion_predictions(
                    y_test_array,
                    np.asarray(
                        qwen_test_predictions
                    ),
                )
            )

            # -------------------
            # Learned fusion
            # -------------------

            fusion_classifier = (
                FusionClassifier(
                    seed=seed
                )
            )

            fusion_classifier.fit(
                np.stack(
                    x_fusion_train
                ),
                y_train_array,
            )

            fusion_predictions = (
                fusion_classifier.predict(
                    np.stack(
                        x_fusion_test
                    )
                )
            )

            fusion_metrics = (
                evaluate_fusion_predictions(
                    y_test_array,
                    fusion_predictions,
                )
            )

            for (
                model_name,
                metrics,
            ) in [
                (
                    "biomedclip",
                    bio_metrics,
                ),
                (
                    "qwen",
                    qwen_metrics,
                ),
                (
                    "fusion",
                    fusion_metrics,
                ),
            ]:
                rows.append(
                    {
                        "seed": seed,

                        "finding": (
                            finding
                        ),

                        "model": (
                            model_name
                        ),

                        "macro_f1": (
                            metrics[
                                "macro_f1"
                            ]
                        ),

                        "balanced_accuracy": (
                            metrics[
                                "balanced_accuracy"
                            ]
                        ),

                        "absent_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "absent"
                            ]
                        ),

                        "new_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "new"
                            ]
                        ),

                        "persistent_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "persistent"
                            ]
                        ),

                        "resolved_recall": (
                            metrics[
                                "state_recall"
                            ][
                                "resolved"
                            ]
                        ),

                        "train_pairs": (
                            len(
                                train_ids
                            )
                        ),

                        "test_pairs": (
                            len(
                                test_ids
                            )
                        ),

                        "patient_overlap": (
                            len(
                                overlap
                            )
                        ),
                    }
                )

    results = pd.DataFrame(
        rows
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir
        / "fusion_seed_results.csv",
        index=False,
    )

    return results
def aggregate_fusion_results(
    results: pd.DataFrame,
) -> pd.DataFrame:

    aggregate = (
        results
        .groupby(
            [
                "finding",
                "model",
            ],
            as_index=False,
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),

            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),

            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),

            new_recall_mean=(
                "new_recall",
                "mean",
            ),

            resolved_recall_mean=(
                "resolved_recall",
                "mean",
            ),

            persistent_recall_mean=(
                "persistent_recall",
                "mean",
            ),

            absent_recall_mean=(
                "absent_recall",
                "mean",
            ),

            num_seeds=(
                "seed",
                "nunique",
            ),
        )
    )

    aggregate[
        "macro_f1_std"
    ] = aggregate[
        "macro_f1_std"
    ].fillna(
        0.0
    )

    aggregate[
        "balanced_accuracy_std"
    ] = aggregate[
        "balanced_accuracy_std"
    ].fillna(
        0.0
    )

    return aggregate
def aggregate_models(
    results: pd.DataFrame,
) -> pd.DataFrame:

    return (
        results
        .groupby(
            "model",
            as_index=False,
        )
        .agg(
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),

            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),

            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
        )
    )