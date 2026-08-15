import pytest

from medchange.inference.structured_output import (
    extract_json,
    parse_vlm_response,
)


def test_extract_plain_json():
    text = """
    {
        "findings": [],
        "impression": "No clear abnormality.",
        "overall_confidence": 0.75,
        "requires_review": true
    }
    """

    data = extract_json(text)

    assert data["overall_confidence"] == 0.75


def test_extract_markdown_json():
    fence = "`" * 3

    text = (
        f"{fence}json\n"
        "{\n"
        '    "findings": [],\n'
        '    "impression": "No acute abnormality.",\n'
        '    "overall_confidence": 0.8,\n'
        '    "requires_review": true\n'
        "}\n"
        f"{fence}"
    )

    data = extract_json(text)

    assert data["impression"] == "No acute abnormality."
    assert data["overall_confidence"] == 0.8


def test_parse_prediction():
    raw = """
    {
        "findings": [
            {
                "name": "pleural_effusion",
                "present": true,
                "confidence": 0.82,
                "anatomy": "right_lower_thorax"
            }
        ],
        "impression": "Possible right pleural effusion.",
        "overall_confidence": 0.72,
        "requires_review": true
    }
    """

    prediction = parse_vlm_response(
        raw_output=raw,
        study_id="study-001",
    )

    assert prediction.study_id == "study-001"
    assert len(prediction.findings) == 1
    assert prediction.findings[0].name == "pleural_effusion"
    assert prediction.findings[0].confidence == pytest.approx(0.82)

    assert prediction.uncertainty is not None
    assert prediction.uncertainty.overall_confidence == pytest.approx(0.72)
    assert prediction.uncertainty.requires_review is True


def test_force_human_review():
    raw = """
    {
        "findings": [],
        "impression": "No abnormality.",
        "overall_confidence": 0.95,
        "requires_review": false
    }
    """

    prediction = parse_vlm_response(
        raw_output=raw,
        study_id="study-002",
    )

    assert prediction.uncertainty is not None
    assert prediction.uncertainty.requires_review is True


def test_extract_json_with_extra_text():
    text = """
    Model analysis:

    {
        "findings": [],
        "impression": "No definite acute abnormality.",
        "overall_confidence": 0.60,
        "requires_review": true
    }

    End of response.
    """

    data = extract_json(text)

    assert data["overall_confidence"] == pytest.approx(0.60)


def test_invalid_json_raises_error():
    raw = "This output contains no valid JSON."

    with pytest.raises(
        ValueError,
        match="No JSON object found",
    ):
        extract_json(raw)