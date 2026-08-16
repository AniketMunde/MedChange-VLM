from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medchange.data.nih.temporal_split import (
    patient_aware_temporal_split,
)
from medchange.evaluation.selective_metrics import (
    compute_selective_metrics,
)
from medchange.fusion.classifier import (
    FusionClassifier,
)
from medchange.fusion.config import (
    BEST_BIOMEDCLIP_FEATURES,
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
from medchange.reasoning.decision import (
    resolve_temporal_decision,
)
from medchange.reasoning.evidence import (
    build_model_evidence,
)
from medchange.safety.policy import (
    apply_safety_policy,
)


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


DEFAULT_SEEDS = [
    11,
    21,
    42,
    84,
    123,
]


def _load_inputs(
    temporal_pairs_path: str | Path,
    qwen_finding_cache_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal = pd.read_csv(
        temporal_pairs_path
    )

    qwen = pd.read_csv(
        qwen_finding_cache_path
    )

    required_temporal = {
        "pair_id",
        "patient_id",
        "prior_image_index",
        "current_image_index",
    }

    for finding in TARGET_FINDINGS:
        required_temporal.add(
            f"{finding}_temporal"
        )

    missing_temporal = (
        required_temporal
        - set(
            temporal.columns
        )
    )

    if missing_temporal:
        raise ValueError(
            "Temporal input missing required columns: "
            f"{sorted(missing_temporal)}"
        )

    required_qwen = {
        "pair_id",
        "finding",
        "qwen_state",
        "qwen_confidence",
    }

    missing_qwen = (
        required_qwen
        - set(
            qwen.columns
        )
    )

    if missing_qwen:
        raise ValueError(
            "Qwen cache missing required columns: "
            f"{sorted(missing_qwen)}"
        )

    temporal = temporal.copy()
    qwen = qwen.copy()

    temporal["pair_id"] = (
        temporal["pair_id"]
        .astype(str)
    )

    temporal["patient_id"] = (
        temporal["patient_id"]
        .astype(str)
    )

    qwen["pair_id"] = (
        qwen["pair_id"]
        .astype(str)
    )

    qwen["finding"] = (
        qwen["finding"]
        .astype(str)
    )

    duplicate_qwen = (
        qwen.duplicated(
            [
                "pair_id",
                "finding",
            ]
        )
        .sum()
    )

    if duplicate_qwen:
        raise ValueError(
            "Duplicate Qwen evidence rows found: "
            f"{duplicate_qwen}"
        )

    qwen_pair_ids = set(
        qwen["pair_id"]
    )

    temporal = temporal[
        temporal["pair_id"]
        .isin(
            qwen_pair_ids
        )
    ].copy()

    if temporal.empty:
        raise ValueError(
            "No temporal pairs matched the Qwen cache."
        )

    return (
        temporal,
        qwen,
    )


def _build_qwen_lookup(
    qwen: pd.DataFrame,
) -> dict[
    tuple[str, str],
    tuple[str, float | None],
]:
    lookup = {}

    for _, row in (
        qwen.iterrows()
    ):
        confidence = (
            None
            if pd.isna(
                row[
                    "qwen_confidence"
                ]
            )
            else float(
                row[
                    "qwen_confidence"
                ]
            )
        )

        lookup[
            (
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
        ] = (
            str(
                row[
                    "qwen_state"
                ]
            ),
            confidence,
        )

    return lookup


def _build_temporal_feature_cache(
    temporal: pd.DataFrame,
    embedding_cache_dir: str | Path,
) -> dict[str, Any]:
    """
    Build temporal embedding feature objects once.

    These are reused across all five patient-aware seeds.
    """

    cache = EmbeddingCache(
        embedding_cache_dir
    )

    feature_cache = {}

    total = len(
        temporal
    )

    for position, (_, row) in enumerate(
        temporal.iterrows(),
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
                "Missing BiomedCLIP embedding: "
                f"{prior_index}"
            )

        if not cache.contains(
            current_index
        ):
            raise FileNotFoundError(
                "Missing BiomedCLIP embedding: "
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

        feature_cache[
            pair_id
        ] = (
            build_temporal_embedding_features(
                prior_embedding,
                current_embedding,
            )
        )

        if (
            position == 1
            or position % 25 == 0
            or position == total
        ):
            print(
                f"[{position}/{total}] "
                "temporal features prepared"
            )

    return feature_cache


def _build_matrix(
    dataframe: pd.DataFrame,
    finding: str,
    feature_set: str,
    feature_cache: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    vectors = []
    labels = []

    target_column = (
        f"{finding}_temporal"
    )

    for _, row in (
        dataframe.iterrows()
    ):
        pair_id = str(
            row[
                "pair_id"
            ]
        )

        temporal_features = (
            feature_cache[
                pair_id
            ]
        )

        vector = (
            build_ablation_vector(
                temporal_features,
                feature_set,
            )
        )

        vectors.append(
            vector
        )

        labels.append(
            str(
                row[
                    target_column
                ]
            )
        )

    if not vectors:
        raise ValueError(
            f"No samples available for {finding}."
        )

    return (
        np.stack(
            vectors
        ),
        np.asarray(
            labels
        ),
    )


def _train_biomedclip_classifier(
    train_dataframe: pd.DataFrame,
    finding: str,
    feature_cache: dict[str, Any],
    seed: int,
) -> FusionClassifier:
    feature_set = (
        BEST_BIOMEDCLIP_FEATURES[
            finding
        ]
    )

    x_train, y_train = (
        _build_matrix(
            dataframe=(
                train_dataframe
            ),
            finding=(
                finding
            ),
            feature_set=(
                feature_set
            ),
            feature_cache=(
                feature_cache
            ),
        )
    )

    classes = sorted(
        set(
            y_train.tolist()
        )
    )

    if len(classes) < 2:
        raise ValueError(
            f"Training split for {finding} "
            "contains fewer than two classes: "
            f"{classes}"
        )

    classifier = (
        FusionClassifier(
            seed=seed
        )
    )

    classifier.fit(
        x_train,
        y_train,
    )

    return classifier


def _predict_biomedclip(
    classifier: FusionClassifier,
    temporal_features: Any,
    finding: str,
) -> tuple[
    str,
    float | None,
]:
    feature_set = (
        BEST_BIOMEDCLIP_FEATURES[
            finding
        ]
    )

    feature_vector = (
        build_ablation_vector(
            temporal_features,
            feature_set,
        )
        .reshape(
            1,
            -1,
        )
    )

    state = str(
        classifier.predict(
            feature_vector
        )[0]
    )

    confidence = None

    if (
        hasattr(
            classifier,
            "model",
        )
        and hasattr(
            classifier.model,
            "predict_proba",
        )
    ):
        scaled = (
            classifier.scaler
            .transform(
                feature_vector
            )
        )

        probabilities = (
            classifier.model
            .predict_proba(
                scaled
            )[0]
        )

        confidence = float(
            np.max(
                probabilities
            )
        )

    return (
        state,
        confidence,
    )


def _run_one_seed(
    temporal: pd.DataFrame,
    qwen_lookup: dict,
    feature_cache: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    print()
    print("=" * 100)
    print(
        f"M5.5.1 patient-aware evaluation "
        f"seed={seed}"
    )
    print("=" * 100)

    split = (
        patient_aware_temporal_split(
            temporal,
            train_fraction=0.70,
            validation_fraction=0.15,
            seed=seed,
        )
    )

    train = (
        split.train.copy()
    )

    test = (
        split.test.copy()
    )

    train[
        "patient_id"
    ] = train[
        "patient_id"
    ].astype(str)

    test[
        "patient_id"
    ] = test[
        "patient_id"
    ].astype(str)

    train_patients = set(
        train[
            "patient_id"
        ]
    )

    test_patients = set(
        test[
            "patient_id"
        ]
    )

    overlap = (
        train_patients
        & test_patients
    )

    if overlap:
        raise RuntimeError(
            "Patient leakage detected "
            f"for seed {seed}: "
            f"{len(overlap)} overlapping patients."
        )

    print(
        f"Train pairs       : "
        f"{len(train)}"
    )

    print(
        f"Test pairs        : "
        f"{len(test)}"
    )

    print(
        f"Train patients    : "
        f"{len(train_patients)}"
    )

    print(
        f"Test patients     : "
        f"{len(test_patients)}"
    )

    print(
        f"Patient overlap   : "
        f"{len(overlap)}"
    )

    classifiers = {}

    for finding in (
        TARGET_FINDINGS
    ):
        print(
            f"Training "
            f"{finding:<20} "
            f"features="
            f"{BEST_BIOMEDCLIP_FEATURES[finding]}"
        )

        classifiers[
            finding
        ] = (
            _train_biomedclip_classifier(
                train_dataframe=train,
                finding=finding,
                feature_cache=(
                    feature_cache
                ),
                seed=seed,
            )
        )

    rows = []

    for _, pair in (
        test.iterrows()
    ):
        pair_id = str(
            pair[
                "pair_id"
            ]
        )

        patient_id = str(
            pair[
                "patient_id"
            ]
        )

        temporal_features = (
            feature_cache[
                pair_id
            ]
        )

        for finding in (
            TARGET_FINDINGS
        ):
            ground_truth = str(
                pair[
                    f"{finding}_temporal"
                ]
            )

            (
                bio_state,
                bio_confidence,
            ) = (
                _predict_biomedclip(
                    classifier=(
                        classifiers[
                            finding
                        ]
                    ),
                    temporal_features=(
                        temporal_features
                    ),
                    finding=(
                        finding
                    ),
                )
            )

            qwen_key = (
                pair_id,
                finding,
            )

            if (
                qwen_key
                in qwen_lookup
            ):
                (
                    qwen_state,
                    qwen_confidence,
                ) = (
                    qwen_lookup[
                        qwen_key
                    ]
                )

            else:
                qwen_state = (
                    "uncertain"
                )

                qwen_confidence = (
                    None
                )

            bio_evidence = (
                build_model_evidence(
                    state=(
                        bio_state
                    ),
                    confidence=(
                        bio_confidence
                    ),
                )
            )

            qwen_evidence = (
                build_model_evidence(
                    state=(
                        qwen_state
                    ),
                    confidence=(
                        qwen_confidence
                    ),
                )
            )

            decision = (
                resolve_temporal_decision(
                    biomedclip=(
                        bio_evidence
                    ),
                    qwen=(
                        qwen_evidence
                    ),
                )
            )

            safety = (
                apply_safety_policy(
                    final_state=(
                        decision.final_state
                    ),
                    agreement=(
                        decision.agreement
                    ),
                    biomedclip_confidence=(
                        bio_confidence
                    ),
                    qwen_confidence=(
                        qwen_confidence
                    ),
                )
            )

            rows.append(
                {
                    "seed": (
                        seed
                    ),

                    "pair_id": (
                        pair_id
                    ),

                    "patient_id": (
                        patient_id
                    ),

                    "finding": (
                        finding
                    ),

                    "ground_truth": (
                        ground_truth
                    ),

                    "biomedclip_state": (
                        bio_state
                    ),

                    "biomedclip_confidence": (
                        bio_confidence
                    ),

                    "qwen_state": (
                        qwen_state
                    ),

                    "qwen_confidence": (
                        qwen_confidence
                    ),

                    "medchange_state": (
                        safety.final_state
                    ),

                    "agreement": (
                        decision.agreement
                    ),

                    "uncertainty": (
                        safety.uncertainty
                    ),

                    "requires_review": (
                        safety.requires_review
                    ),

                    "abstained": (
                        safety.abstained
                    ),

                    "decision_reason": (
                        decision.reason
                    ),

                    "safety_reason": (
                        safety.reason
                    ),

                    "train_pairs": (
                        len(
                            train
                        )
                    ),

                    "test_pairs": (
                        len(
                            test
                        )
                    ),

                    "train_patients": (
                        len(
                            train_patients
                        )
                    ),

                    "test_patients": (
                        len(
                            test_patients
                        )
                    ),

                    "patient_overlap": (
                        len(
                            overlap
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def _evaluate_prediction_column(
    dataframe: pd.DataFrame,
    prediction_column: str,
    review_column: str | None = None,
) -> dict[str, Any]:
    y_true = (
        dataframe[
            "ground_truth"
        ]
        .astype(str)
        .to_numpy()
    )

    y_pred = (
        dataframe[
            prediction_column
        ]
        .astype(str)
        .to_numpy()
    )

    if review_column is None:
        review = np.zeros(
            len(
                dataframe
            ),
            dtype=bool,
        )

    else:
        review = (
            dataframe[
                review_column
            ]
            .astype(bool)
            .to_numpy()
        )

    return compute_selective_metrics(
        y_true=y_true,
        y_pred=y_pred,
        requires_review=review,
    )


def evaluate_seed(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "biomedclip":
            _evaluate_prediction_column(
                dataframe,
                "biomedclip_state",
            ),

        "qwen":
            _evaluate_prediction_column(
                dataframe,
                "qwen_state",
            ),

        "medchange":
            _evaluate_prediction_column(
                dataframe,
                "medchange_state",
                "requires_review",
            ),
    }


def compute_pair_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    pair_rows = []

    for (
        pair_id,
        group,
    ) in (
        dataframe.groupby(
            "pair_id"
        )
    ):
        true_states = (
            group[
                "ground_truth"
            ]
            .astype(str)
            .tolist()
        )

        medchange_states = (
            group[
                "medchange_state"
            ]
            .astype(str)
            .tolist()
        )

        covered_mask = (
            group[
                "medchange_state"
            ]
            .astype(str)
            != "uncertain"
        )

        all_findings_covered = bool(
            covered_mask.all()
        )

        exact_match = bool(
            all_findings_covered
            and (
                true_states
                == medchange_states
            )
        )

        pair_rows.append(
            {
                "pair_id": (
                    pair_id
                ),

                "exact_match": (
                    exact_match
                ),

                "any_abstention": bool(
                    group[
                        "abstained"
                    ]
                    .astype(bool)
                    .any()
                ),

                "any_review": bool(
                    group[
                        "requires_review"
                    ]
                    .astype(bool)
                    .any()
                ),

                "fully_covered": (
                    all_findings_covered
                ),

                "covered_findings": int(
                    covered_mask.sum()
                ),

                "total_findings": int(
                    len(
                        group
                    )
                ),
            }
        )

    pair_frame = pd.DataFrame(
        pair_rows
    )

    return {
        "num_pairs": int(
            len(
                pair_frame
            )
        ),

        "exact_pair_match_rate": float(
            pair_frame[
                "exact_match"
            ].mean()
        ),

        "pair_abstention_rate": float(
            pair_frame[
                "any_abstention"
            ].mean()
        ),

        "pair_review_rate": float(
            pair_frame[
                "any_review"
            ].mean()
        ),

        "fully_covered_pair_rate": float(
            pair_frame[
                "fully_covered"
            ].mean()
        ),

        "mean_covered_findings_per_pair": float(
            pair_frame[
                "covered_findings"
            ].mean()
        ),
    }


def _flatten_seed_metrics(
    seed: int,
    metrics: dict[
        str,
        Any,
    ],
) -> list[dict]:
    rows = []

    for (
        model,
        model_metrics,
    ) in (
        metrics.items()
    ):
        rows.append(
            {
                "seed": (
                    seed
                ),

                "model": (
                    model
                ),

                "coverage": (
                    model_metrics[
                        "coverage"
                    ]
                ),

                "abstention_rate": (
                    model_metrics[
                        "abstention_rate"
                    ]
                ),

                "selective_accuracy": (
                    model_metrics[
                        "selective_accuracy"
                    ]
                ),

                "selective_macro_f1": (
                    model_metrics[
                        "selective_macro_f1"
                    ]
                ),

                "error_rate_on_covered": (
                    model_metrics[
                        "error_rate_on_covered"
                    ]
                ),

                "review_rate": (
                    model_metrics[
                        "review_rate"
                    ]
                ),

                "absent_recall": (
                    model_metrics[
                        "state_recall"
                    ][
                        "absent"
                    ]
                ),

                "new_recall": (
                    model_metrics[
                        "state_recall"
                    ][
                        "new"
                    ]
                ),

                "persistent_recall": (
                    model_metrics[
                        "state_recall"
                    ][
                        "persistent"
                    ]
                ),

                "resolved_recall": (
                    model_metrics[
                        "state_recall"
                    ][
                        "resolved"
                    ]
                ),
            }
        )

    return rows


def aggregate_model_metrics(
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        "coverage",
        "abstention_rate",
        "selective_accuracy",
        "selective_macro_f1",
        "error_rate_on_covered",
        "review_rate",
        "absent_recall",
        "new_recall",
        "persistent_recall",
        "resolved_recall",
    ]

    rows = []

    for model in sorted(
        seed_metrics[
            "model"
        ].unique()
    ):
        subset = seed_metrics[
            seed_metrics[
                "model"
            ]
            == model
        ]

        output = {
            "model": model,
            "num_seeds": int(
                subset[
                    "seed"
                ].nunique()
            ),
        }

        for column in (
            numeric_columns
        ):
            values = pd.to_numeric(
                subset[
                    column
                ],
                errors="coerce",
            )

            valid = values.dropna()

            if valid.empty:
                output[
                    f"{column}_mean"
                ] = None

                output[
                    f"{column}_std"
                ] = None

            else:
                output[
                    f"{column}_mean"
                ] = float(
                    valid.mean()
                )

                output[
                    f"{column}_std"
                ] = float(
                    valid.std(
                        ddof=1
                    )
                    if len(
                        valid
                    ) > 1
                    else 0.0
                )

        rows.append(
            output
        )

    return pd.DataFrame(
        rows
    )


def aggregate_pair_metrics(
    pair_seed_rows: list[dict],
) -> dict[str, Any]:
    dataframe = pd.DataFrame(
        pair_seed_rows
    )

    numeric_columns = [
        "exact_pair_match_rate",
        "pair_abstention_rate",
        "pair_review_rate",
        "fully_covered_pair_rate",
        "mean_covered_findings_per_pair",
    ]

    result = {
        "num_seeds": int(
            dataframe[
                "seed"
            ].nunique()
        )
    }

    for column in numeric_columns:
        values = (
            dataframe[
                column
            ]
            .astype(float)
        )

        result[
            f"{column}_mean"
        ] = float(
            values.mean()
        )

        result[
            f"{column}_std"
        ] = float(
            values.std(
                ddof=1
            )
            if len(
                values
            ) > 1
            else 0.0
        )

    return result


def run_medchange_system_benchmark(
    temporal_pairs_path: str | Path,
    qwen_finding_cache_path: str | Path,
    embedding_cache_dir: str | Path,
    output_dir: str | Path,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """
    Patient-aware out-of-fold MedChange evaluation.

    IMPORTANT:
    No deployment classifier artifacts are used here.

    For every seed:
      1. split by patient
      2. train BiomedCLIP classifiers on train patients
      3. evaluate BiomedCLIP on unseen test patients
      4. replay cached Qwen predictions on those same test pairs
      5. run MedChange disagreement + safety policy
      6. calculate selective metrics
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        temporal,
        qwen,
    ) = _load_inputs(
        temporal_pairs_path=(
            temporal_pairs_path
        ),
        qwen_finding_cache_path=(
            qwen_finding_cache_path
        ),
    )

    print()
    print("=" * 100)
    print(
        "Preparing cached temporal features"
    )
    print("=" * 100)

    feature_cache = (
        _build_temporal_feature_cache(
            temporal=temporal,
            embedding_cache_dir=(
                embedding_cache_dir
            ),
        )
    )

    qwen_lookup = (
        _build_qwen_lookup(
            qwen
        )
    )

    all_predictions = []

    seed_metric_rows = []

    pair_seed_rows = []

    audit_rows = []

    for seed in seeds:
        seed_predictions = (
            _run_one_seed(
                temporal=temporal,
                qwen_lookup=(
                    qwen_lookup
                ),
                feature_cache=(
                    feature_cache
                ),
                seed=seed,
            )
        )

        all_predictions.append(
            seed_predictions
        )

        seed_metrics = (
            evaluate_seed(
                seed_predictions
            )
        )

        seed_metric_rows.extend(
            _flatten_seed_metrics(
                seed=seed,
                metrics=(
                    seed_metrics
                ),
            )
        )

        pair_metrics = (
            compute_pair_metrics(
                seed_predictions
            )
        )

        pair_seed_rows.append(
            {
                "seed": seed,
                **pair_metrics,
            }
        )

        audit_rows.append(
            {
                "seed": seed,

                "train_pairs": int(
                    seed_predictions[
                        "train_pairs"
                    ]
                    .iloc[0]
                ),

                "test_pairs": int(
                    seed_predictions[
                        "test_pairs"
                    ]
                    .iloc[0]
                ),

                "train_patients": int(
                    seed_predictions[
                        "train_patients"
                    ]
                    .iloc[0]
                ),

                "test_patients": int(
                    seed_predictions[
                        "test_patients"
                    ]
                    .iloc[0]
                ),

                "patient_overlap": int(
                    seed_predictions[
                        "patient_overlap"
                    ]
                    .iloc[0]
                ),
            }
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    seed_metrics_frame = (
        pd.DataFrame(
            seed_metric_rows
        )
    )

    pair_metrics_frame = (
        pd.DataFrame(
            pair_seed_rows
        )
    )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    if (
        audit_frame[
            "patient_overlap"
        ]
        .sum()
        != 0
    ):
        raise RuntimeError(
            "Patient leakage detected in "
            "M5.5.1 evaluation."
        )

    model_aggregate = (
        aggregate_model_metrics(
            seed_metrics_frame
        )
    )

    pair_aggregate = (
        aggregate_pair_metrics(
            pair_seed_rows
        )
    )

    conflict_rate = float(
        (
            predictions[
                "agreement"
            ]
            == "conflict"
        ).mean()
    )

    uncertainty_rate = float(
        (
            predictions[
                "uncertainty"
            ]
            != "low"
        ).mean()
    )

    predictions.to_csv(
        output_dir
        / "oof_system_predictions.csv",
        index=False,
    )

    seed_metrics_frame.to_csv(
        output_dir
        / "seed_model_metrics.csv",
        index=False,
    )

    pair_metrics_frame.to_csv(
        output_dir
        / "seed_pair_metrics.csv",
        index=False,
    )

    model_aggregate.to_csv(
        output_dir
        / "model_aggregate.csv",
        index=False,
    )

    audit_frame.to_csv(
        output_dir
        / "split_audit.csv",
        index=False,
    )

    results = {
        "evaluation": (
            "patient-aware out-of-fold"
        ),

        "seeds": (
            seeds
        ),

        "num_source_pairs": int(
            len(
                temporal
            )
        ),

        "patient_overlap_all_seeds": int(
            audit_frame[
                "patient_overlap"
            ]
            .sum()
        ),

        "models": (
            model_aggregate
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),

        "pair_metrics": (
            pair_aggregate
        ),

        "conflict_rate": (
            conflict_rate
        ),

        "uncertainty_rate": (
            uncertainty_rate
        ),
    }

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    return results