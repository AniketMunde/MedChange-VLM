from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


VALID_TEMPORAL_STATES = {
    "absent",
    "new",
    "persistent",
    "resolved",
}


SYSTEM_PROMPT = (
    "You are MedChange-VLM, a research assistant for "
    "longitudinal chest X-ray comparison. "
    "Compare a prior chest radiograph with a current "
    "chest radiograph. "
    "For each supported finding, classify temporal change "
    "as exactly one of: absent, new, persistent, resolved. "
    "Return only valid JSON. "
    "Do not introduce findings outside the supported list."
)


USER_PROMPT = (
    "The first image is the PRIOR chest radiograph and the "
    "second image is the CURRENT chest radiograph.\n\n"
    "Compare them for these seven findings:\n"
    "- atelectasis\n"
    "- cardiomegaly\n"
    "- consolidation\n"
    "- edema\n"
    "- pleural_effusion\n"
    "- pneumonia\n"
    "- pneumothorax\n\n"
    "For every finding, return its temporal state using only:\n"
    "absent, new, persistent, resolved.\n\n"
    "Return JSON with this structure:\n"
    "{\n"
    '  "findings": [\n'
    '    {"finding": "atelectasis", "change": "..."},\n'
    "    ...\n"
    "  ],\n"
    '  "overall_change": "..."\n'
    "}"
)


def normalize_state(
    value: Any,
) -> str:
    state = (
        str(value)
        .strip()
        .lower()
    )

    if (
        state
        not in VALID_TEMPORAL_STATES
    ):
        raise ValueError(
            f"Unsupported temporal state: {value}"
        )

    return state


def derive_overall_change(
    states: list[str],
) -> str:
    has_new = (
        "new"
        in states
    )

    has_resolved = (
        "resolved"
        in states
    )

    if (
        has_new
        and has_resolved
    ):
        return "mixed"

    if has_new:
        return "worsened"

    if has_resolved:
        return "improved"

    return "stable"


def build_target_payload(
    row: pd.Series,
) -> dict[str, Any]:
    findings = []

    states = []

    for finding in (
        TARGET_FINDINGS
    ):
        column = (
            f"{finding}_temporal"
        )

        if column not in row.index:
            raise ValueError(
                f"Missing temporal column: {column}"
            )

        state = normalize_state(
            row[
                column
            ]
        )

        states.append(
            state
        )

        findings.append(
            {
                "finding": finding,
                "change": state,
            }
        )

    return {
        "findings": findings,
        "overall_change":
            derive_overall_change(
                states
            ),
    }


def build_sft_record(
    *,
    row: pd.Series,
    prior_image_path: str,
    current_image_path: str,
) -> dict[str, Any]:
    target = (
        build_target_payload(
            row
        )
    )

    target_text = json.dumps(
        target,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )

    return {
        "pair_id": str(
            row[
                "pair_id"
            ]
        ),

        "patient_id": str(
            row[
                "patient_id"
            ]
        ),

        "images": [
            prior_image_path,
            current_image_path,
        ],

        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            SYSTEM_PROMPT
                        ),
                    }
                ],
            },

            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": (
                            USER_PROMPT
                        ),
                    },
                ],
            },

            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            target_text
                        ),
                    }
                ],
            },
        ],

        "target": target,
    }


def validate_record(
    record: dict[str, Any],
) -> None:
    images = (
        record.get(
            "images",
            []
        )
    )

    if len(
        images
    ) != 2:
        raise ValueError(
            "Each training example must "
            "contain exactly two images."
        )

    messages = (
        record.get(
            "messages",
            []
        )
    )

    if len(
        messages
    ) != 3:
        raise ValueError(
            "Expected system, user and "
            "assistant messages."
        )

    target = (
        record.get(
            "target",
            {}
        )
    )

    findings = (
        target.get(
            "findings",
            []
        )
    )

    if len(
        findings
    ) != len(
        TARGET_FINDINGS
    ):
        raise ValueError(
            "Target must contain exactly "
            "seven findings."
        )

    names = [
        item[
            "finding"
        ]
        for item
        in findings
    ]

    if names != (
        TARGET_FINDINGS
    ):
        raise ValueError(
            "Finding ordering does not "
            "match TARGET_FINDINGS."
        )

    for item in findings:
        normalize_state(
            item[
                "change"
            ]
        )


def patient_disjoint_split(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> dict[
    str,
    pd.DataFrame,
]:
    if not (
        0
        < train_fraction
        < 1
    ):
        raise ValueError(
            "train_fraction must be "
            "between 0 and 1."
        )

    if not (
        0
        <= validation_fraction
        < 1
    ):
        raise ValueError(
            "validation_fraction must be "
            "between 0 and 1."
        )

    if (
        train_fraction
        + validation_fraction
        >= 1
    ):
        raise ValueError(
            "train + validation fractions "
            "must be less than 1."
        )

    patient_ids = (
        dataframe[
            "patient_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        patient_ids
    )

    total_patients = len(
        patient_ids
    )

    train_end = int(
        total_patients
        * train_fraction
    )

    validation_end = (
        train_end
        + int(
            total_patients
            * validation_fraction
        )
    )

    train_patients = set(
        patient_ids[
            :train_end
        ]
    )

    validation_patients = set(
        patient_ids[
            train_end:
            validation_end
        ]
    )

    test_patients = set(
        patient_ids[
            validation_end:
        ]
    )

    patient_series = (
        dataframe[
            "patient_id"
        ]
        .astype(str)
    )

    train = dataframe[
        patient_series.isin(
            train_patients
        )
    ].copy()

    validation = dataframe[
        patient_series.isin(
            validation_patients
        )
    ].copy()

    test = dataframe[
        patient_series.isin(
            test_patients
        )
    ].copy()

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

    if overlap:
        raise RuntimeError(
            "Patient leakage detected."
        )

    return {
        "train": (
            train
        ),

        "validation": (
            validation
        ),

        "test": (
            test
        ),
    }


def write_jsonl(
    records: list[
        dict[str, Any]
    ],
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            validate_record(
                record
            )

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )