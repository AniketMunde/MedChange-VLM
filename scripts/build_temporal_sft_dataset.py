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


from medchange.training.temporal_sft_dataset import (
    TARGET_FINDINGS,
    build_sft_record,
    patient_disjoint_split,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build patient-disjoint "
            "MedChange QLoRA temporal "
            "instruction datasets."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/nih/"
            "temporal_dataset.csv"
        ),
    )

    parser.add_argument(
        "--image-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/nih/qlora"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
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

    for _, row in (
        dataframe.iterrows()
    ):
        prior_path = (
            resolve_image(
                image_root,
                str(
                    row[
                        "prior_image_index"
                    ]
                ),
            )
        )

        current_path = (
            resolve_image(
                image_root,
                str(
                    row[
                        "current_image_index"
                    ]
                ),
            )
        )

        record = (
            build_sft_record(
                row=row,

                prior_image_path=(
                    str(
                        prior_path.resolve()
                    )
                ),

                current_image_path=(
                    str(
                        current_path.resolve()
                    )
                ),
            )
        )

        records.append(
            record
        )

    return records


def summarize_split(
    name: str,
    dataframe: pd.DataFrame,
) -> dict:
    summary = {
        "split": name,

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

        "states": {},
    }

    for finding in (
        TARGET_FINDINGS
    ):
        column = (
            f"{finding}_temporal"
        )

        summary[
            "states"
        ][
            finding
        ] = {
            str(
                key
            ): int(
                value
            )

            for (
                key,
                value,
            )
            in (
                dataframe[
                    column
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        }

    return summary


def main() -> None:
    args = (
        parse_args()
    )

    input_path = Path(
        args.input
    )

    image_root = Path(
        args.image_root
    )

    output_dir = Path(
        args.output_dir
    )

    dataframe = pd.read_csv(
        input_path
    )

    if args.max_samples:
        dataframe = (
            dataframe
            .head(
                args.max_samples
            )
            .copy()
        )

    splits = (
        patient_disjoint_split(
            dataframe,

            train_fraction=(
                args.train_fraction
            ),

            validation_fraction=(
                args.validation_fraction
            ),

            seed=(
                args.seed
            ),
        )
    )

    summaries = []

    split_patient_sets = {}

    for (
        split_name,
        split_df,
    ) in (
        splits.items()
    ):
        print()
        print(
            f"Building "
            f"{split_name} split..."
        )

        records = (
            dataframe_to_records(
                split_df,

                image_root=(
                    image_root
                ),
            )
        )

        output_path = (
            output_dir
            / (
                f"{split_name}.jsonl"
            )
        )

        write_jsonl(
            records,
            output_path,
        )

        summary = (
            summarize_split(
                split_name,
                split_df,
            )
        )

        summaries.append(
            summary
        )

        split_patient_sets[
            split_name
        ] = set(
            split_df[
                "patient_id"
            ]
            .astype(str)
        )

        print(
            f"Samples  : "
            f"{summary['samples']}"
        )

        print(
            f"Patients : "
            f"{summary['patients']}"
        )

        print(
            f"Output   : "
            f"{output_path}"
        )

    train_patients = (
        split_patient_sets[
            "train"
        ]
    )

    validation_patients = (
        split_patient_sets[
            "validation"
        ]
    )

    test_patients = (
        split_patient_sets[
            "test"
        ]
    )

    overlap = (
        train_patients
        & validation_patients
    ) | (
        train_patients
        & test_patients
    ) | (
        validation_patients
        & test_patients
    )

    audit = {
        "seed": (
            args.seed
        ),

        "train_fraction": (
            args.train_fraction
        ),

        "validation_fraction": (
            args.validation_fraction
        ),

        "patient_overlap_count": (
            len(
                overlap
            )
        ),

        "splits": summaries,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        "M7.1 — TEMPORAL SFT DATASET"
    )

    print(
        "=" * 90
    )

    for summary in summaries:
        print(
            f"{summary['split']:<12} "
            f"samples="
            f"{summary['samples']:<8} "
            f"patients="
            f"{summary['patients']}"
        )

    print()

    print(
        "Patient overlap : "
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