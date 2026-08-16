from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.fusion.classifier import (
    FusionClassifier,
)
from medchange.fusion.config import (
    BEST_BIOMEDCLIP_FEATURES,
    FUSION_FINDINGS,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and export final BiomedCLIP "
            "temporal classifiers for MedChange."
        )
    )

    parser.add_argument(
        "--pairs",
        default=(
            "data/nih/"
            "fusion_qwen_subset_200.csv"
        ),
    )

    parser.add_argument(
        "--embedding-cache",
        default=(
            "data/nih/"
            "embedding_cache"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "models/"
            "temporal"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def build_feature_matrix(
    dataframe: pd.DataFrame,
    finding: str,
    feature_set: str,
    cache: EmbeddingCache,
) -> np.ndarray:
    vectors = []

    total = len(
        dataframe
    )

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
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
                "Missing prior embedding: "
                f"{prior_index}"
            )

        if not cache.contains(
            current_index
        ):
            raise FileNotFoundError(
                "Missing current embedding: "
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

        vector = (
            build_ablation_vector(
                temporal_features,
                feature_set,
            )
        )

        vectors.append(
            vector
        )

        if (
            position == 1
            or position % 50 == 0
            or position == total
        ):
            print(
                f"  [{position}/{total}] "
                f"{finding} features prepared"
            )

    return np.stack(
        vectors
    )


def validate_training_labels(
    labels: np.ndarray,
    finding: str,
) -> None:
    unique = sorted(
        set(
            labels.tolist()
        )
    )

    if len(unique) < 2:
        raise ValueError(
            f"{finding} has fewer than "
            "two temporal classes."
        )

    print(
        "  classes:",
        {
            label: int(
                np.sum(
                    labels == label
                )
            )
            for label in unique
        },
    )


def main() -> None:
    args = parse_args()

    pairs_path = Path(
        args.pairs
    )

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Pair dataset not found: "
            f"{pairs_path}"
        )

    dataframe = pd.read_csv(
        pairs_path
    )

    required = {
        "pair_id",
        "patient_id",
        "prior_image_index",
        "current_image_index",
    }

    for finding in (
        FUSION_FINDINGS
    ):
        required.add(
            f"{finding}_temporal"
        )

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Temporal dataset missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    cache = EmbeddingCache(
        args.embedding_cache
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "seed": args.seed,
        "training_pairs": int(
            len(
                dataframe
            )
        ),
        "unique_patients": int(
            dataframe[
                "patient_id"
            ].nunique()
        ),
        "findings": {},
    }

    print()
    print("=" * 90)
    print(
        "M5.1.1 — EXPORT TEMPORAL CLASSIFIERS"
    )
    print("=" * 90)

    print(
        f"Training pairs : "
        f"{len(dataframe)}"
    )

    print(
        f"Patients       : "
        f"{dataframe['patient_id'].nunique()}"
    )

    for finding in (
        FUSION_FINDINGS
    ):
        print()
        print("-" * 90)

        feature_set = (
            BEST_BIOMEDCLIP_FEATURES[
                finding
            ]
        )

        print(
            f"Finding      : "
            f"{finding}"
        )

        print(
            f"Feature set  : "
            f"{feature_set}"
        )

        x = build_feature_matrix(
            dataframe=dataframe,
            finding=finding,
            feature_set=feature_set,
            cache=cache,
        )

        target_column = (
            f"{finding}_temporal"
        )

        y = (
            dataframe[
                target_column
            ]
            .astype(str)
            .to_numpy()
        )

        validate_training_labels(
            y,
            finding,
        )

        classifier = (
            FusionClassifier(
                seed=args.seed
            )
        )

        classifier.fit(
            x,
            y,
        )

        predictions = (
            classifier.predict(
                x
            )
        )

        training_accuracy = float(
            np.mean(
                predictions == y
            )
        )

        artifact = {
            "finding": finding,
            "feature_set": feature_set,
            "classifier": classifier,
            "classes": sorted(
                set(
                    y.tolist()
                )
            ),
            "training_pairs": int(
                len(
                    dataframe
                )
            ),
            "seed": args.seed,
        }

        output_path = (
            output_dir
            / f"{finding}.pkl"
        )

        with output_path.open(
            "wb"
        ) as file:
            pickle.dump(
                artifact,
                file,
            )

        metadata[
            "findings"
        ][
            finding
        ] = {
            "feature_set": (
                feature_set
            ),
            "classes": (
                artifact[
                    "classes"
                ]
            ),
            "feature_dimension": int(
                x.shape[
                    1
                ]
            ),
            "training_accuracy": (
                training_accuracy
            ),
            "artifact": str(
                output_path
            ),
        }

        print(
            f"Feature dim  : "
            f"{x.shape[1]}"
        )

        print(
            f"Train acc    : "
            f"{training_accuracy:.4f}"
        )

        print(
            f"Saved        : "
            f"{output_path}"
        )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print(
        "Export complete."
    )

    print(
        f"Metadata: "
        f"{metadata_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()