from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

if str(
    SRC_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC_DIR
        ),
    )


from medchange.training.change_sampling import (
    category_counts,
    sample_change_aware_pairs,
    state_counts,
)

from medchange.training.temporal_sft_dataset import (
    build_sft_record,
    patient_disjoint_split,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build M7.5 change-aware "
            "patient-disjoint QLoRA dataset."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/nih/"
            "temporal_pairs_same_view.csv"
        ),
    )

    parser.add_argument(
        "--image-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/nih/qlora_m75"
        ),
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def resolve_image(
    image_root: Path,
    image_name: str,
) -> Path:

    path = (
        image_root
        / str(
            image_name
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing image: {path}"
        )

    return path


def dataframe_to_records(
    dataframe: pd.DataFrame,
    *,
    image_root: Path,
) -> list[dict]:

    records = []

    total = len(
        dataframe
    )

    for index, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):

        prior = resolve_image(
            image_root,
            row[
                "prior_image_index"
            ],
        )

        current = resolve_image(
            image_root,
            row[
                "current_image_index"
            ],
        )

        record = (
            build_sft_record(
                row=row,

                prior_image_path=(
                    str(
                        prior.resolve()
                    )
                ),

                current_image_path=(
                    str(
                        current.resolve()
                    )
                ),
            )
        )

        records.append(
            record
        )

        if (
            index == 1
            or index % 500 == 0
            or index == total
        ):
            print(
                f"[{index}/{total}] "
                "records prepared"
            )

    return records


def natural_sample(
    dataframe: pd.DataFrame,
    *,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Validation/test remain natural-distribution
    samples instead of change-balanced samples.
    """

    if len(
        dataframe
    ) <= max_samples:
        return (
            dataframe
            .copy()
        )

    return (
        dataframe
        .sample(
            n=max_samples,
            replace=False,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )


def split_patient_ids(
    dataframe: pd.DataFrame,
) -> set[str]:

    return set(
        dataframe[
            "patient_id"
        ]
        .astype(str)
    )


def main():
    args = parse_args()

    source = pd.read_csv(
        args.input
    )

    print(
        f"Source pairs: "
        f"{len(source):,}"
    )

    print(
        f"Source patients: "
        f"{source['patient_id'].astype(str).nunique():,}"
    )

    # ====================================================
    # PATIENT SPLIT FIRST
    # ====================================================

    split = (
        patient_disjoint_split(
            source,
            train_fraction=0.80,
            validation_fraction=0.10,
            seed=args.seed,
        )
    )

    raw_train = (
        split[
            "train"
        ]
    )

    raw_validation = (
        split[
            "validation"
        ]
    )

    raw_test = (
        split[
            "test"
        ]
    )

    # ====================================================
    # CHANGE-AWARE TRAINING SAMPLE
    # ====================================================

    train = (
        sample_change_aware_pairs(
            raw_train,
            max_samples=(
                args.train_samples
            ),
            seed=args.seed,
        )
    )

    # Validation/test remain natural distribution.
    validation = (
        natural_sample(
            raw_validation,
            max_samples=(
                args.validation_samples
            ),
            seed=(
                args.seed
                + 100
            ),
        )
    )

    test = (
        natural_sample(
            raw_test,
            max_samples=(
                args.test_samples
            ),
            seed=(
                args.seed
                + 200
            ),
        )
    )

    # ====================================================
    # LEAKAGE AUDIT
    # ====================================================

    train_patients = (
        split_patient_ids(
            raw_train
        )
    )

    validation_patients = (
        split_patient_ids(
            raw_validation
        )
    )

    test_patients = (
        split_patient_ids(
            raw_test
        )
    )

    overlap = (
        (
            train_patients
            & validation_patients
        )
        |
        (
            train_patients
            & test_patients
        )
        |
        (
            validation_patients
            & test_patients
        )
    )

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: "
            f"{len(overlap)}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_root = Path(
        args.image_root
    )

    selected = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    audit = {
        "milestone": "M7.5",

        "seed": (
            args.seed
        ),

        "source_pairs": int(
            len(
                source
            )
        ),

        "source_patients": int(
            source[
                "patient_id"
            ]
            .astype(str)
            .nunique()
        ),

        "patient_overlap_count": int(
            len(
                overlap
            )
        ),

        "splits": {},
    }

    for name, dataframe in (
        selected.items()
    ):

        print()
        print(
            "=" * 90
        )

        print(
            name.upper()
        )

        print(
            "=" * 90
        )

        records = (
            dataframe_to_records(
                dataframe,
                image_root=(
                    image_root
                ),
            )
        )

        output_path = (
            output_dir
            / f"{name}.jsonl"
        )

        write_jsonl(
            records,
            output_path,
        )

        audit[
            "splits"
        ][
            name
        ] = {
            "samples": int(
                len(
                    dataframe
                )
            ),

            "patients": int(
                dataframe[
                    "patient_id"
                ]
                .astype(str)
                .nunique()
            ),

            "state_counts":
                state_counts(
                    dataframe
                ),

            "pair_categories":
                category_counts(
                    dataframe
                ),
        }

        print(
            f"Samples    : "
            f"{len(dataframe)}"
        )

        print(
            f"Patients   : "
            f"{dataframe['patient_id'].astype(str).nunique()}"
        )

        print(
            "States     : "
            f"{state_counts(dataframe)}"
        )

        print(
            "Categories : "
            f"{category_counts(dataframe)}"
        )

        print(
            f"Output     : "
            f"{output_path}"
        )

    audit_path = (
        output_dir
        / "dataset_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 90
    )

    print(
        "M7.5 DATASET COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"Patient overlap : "
        f"{len(overlap)}"
    )

    print(
        f"Audit           : "
        f"{audit_path}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()