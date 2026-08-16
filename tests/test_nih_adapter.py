import pytest
from PIL import Image

from medchange.data.nih.adapter import (
    adapt_nih_example,
    convert_nih_labels,
    normalize_raw_labels,
)


def test_normalize_label_list():
    labels = normalize_raw_labels(
        [
            "Effusion",
            "Pneumonia",
        ]
    )

    assert labels == (
        "Effusion",
        "Pneumonia",
    )


def test_convert_nih_labels():
    labels = convert_nih_labels(
        [
            "Effusion",
            "Pneumonia",
        ]
    )

    assert (
        labels[
            "pleural_effusion"
        ]
        == 1
    )

    assert (
        labels[
            "pneumonia"
        ]
        == 1
    )

    assert (
        labels[
            "pneumothorax"
        ]
        == 0
    )


def test_no_finding():
    labels = convert_nih_labels(
        ["No Finding"]
    )

    assert all(
        value == 0
        for value in labels.values()
    )


def test_adapt_nih_example():
    sample = {
        "image": Image.new(
            "L",
            (1024, 1024),
            color=128,
        ),

        "label": [
            "Atelectasis",
            "Effusion",
        ],

        "Patient ID": 123,

        "View Position": "PA",

        "Patient Age": 55,

        "Patient Gender": "M",
    }

    example = (
        adapt_nih_example(
            sample
        )
    )

    assert (
        example.patient_id
        == "123"
    )

    assert (
        example.view_position
        == "PA"
    )

    assert (
        example.image.mode
        == "RGB"
    )

    assert (
        example.labels[
            "atelectasis"
        ]
        == 1
    )

    assert (
        example.labels[
            "pleural_effusion"
        ]
        == 1
    )


def test_missing_required_field():
    sample = {
        "image": Image.new(
            "RGB",
            (64, 64),
        ),
        "label": [],
    }

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        adapt_nih_example(
            sample
        )