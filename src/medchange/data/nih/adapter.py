from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from medchange.data.nih.constants import (
    MEDCHANGE_TO_NIH,
    TARGET_FINDINGS,
)


@dataclass(frozen=True)
class NIHExample:
    """
    Canonical representation of one streamed NIH CXR.
    """

    image: Image.Image

    labels: dict[str, int]

    raw_labels: tuple[str, ...]

    patient_id: str

    view_position: str | None

    patient_age: int | None

    patient_gender: str | None


def normalize_raw_labels(
    labels: object,
) -> tuple[str, ...]:
    """
    Convert the Hugging Face label field to a clean tuple.
    """

    if labels is None:
        return tuple()

    if isinstance(
        labels,
        str,
    ):
        return (
            labels.strip(),
        )

    if isinstance(
        labels,
        (list, tuple),
    ):
        return tuple(
            str(label).strip()
            for label in labels
            if str(label).strip()
        )

    raise TypeError(
        "NIH label field must be a string, list, "
        f"tuple or None. Received: {type(labels)}"
    )


def convert_nih_labels(
    raw_labels: object,
) -> dict[str, int]:
    """
    Convert NIH multi-label annotations into MedChange binary labels.

    Example
    -------
    ["Effusion", "Pneumonia"]

    becomes

    {
        ...
        "pleural_effusion": 1,
        "pneumonia": 1,
        ...
    }
    """

    normalized = set(
        normalize_raw_labels(
            raw_labels
        )
    )

    labels: dict[str, int] = {}

    for medchange_label in TARGET_FINDINGS:
        nih_label = MEDCHANGE_TO_NIH[
            medchange_label
        ]

        labels[
            medchange_label
        ] = int(
            nih_label in normalized
        )

    return labels


def _optional_int(
    value: object,
) -> int | None:

    if value is None:
        return None

    try:
        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def _optional_string(
    value: object,
) -> str | None:

    if value is None:
        return None

    text = str(
        value
    ).strip()

    return text or None


def adapt_nih_example(
    sample: dict,
) -> NIHExample:
    """
    Convert one Hugging Face NIH row to NIHExample.
    """

    required = {
        "image",
        "label",
        "Patient ID",
    }

    missing = (
        required
        - set(sample)
    )

    if missing:
        raise ValueError(
            "NIH sample missing required fields: "
            f"{sorted(missing)}"
        )

    image = sample[
        "image"
    ]

    if not isinstance(
        image,
        Image.Image,
    ):
        raise TypeError(
            "Expected Hugging Face image field "
            "to decode to PIL.Image.Image."
        )

    raw_labels = normalize_raw_labels(
        sample["label"]
    )

    labels = convert_nih_labels(
        raw_labels
    )

    return NIHExample(
        image=image.convert("RGB"),

        labels=labels,

        raw_labels=raw_labels,

        patient_id=str(
            sample["Patient ID"]
        ),

        view_position=_optional_string(
            sample.get(
                "View Position"
            )
        ),

        patient_age=_optional_int(
            sample.get(
                "Patient Age"
            )
        ),

        patient_gender=_optional_string(
            sample.get(
                "Patient Gender"
            )
        ),
    )