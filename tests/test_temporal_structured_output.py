from medchange.inference.temporal_structured_output import (
    parse_temporal_vlm_response,
)


def test_parse_temporal_vlm_response():

    raw = """
    {
      "findings": [
        {
          "finding": "pneumothorax",
          "prior_status": "present",
          "current_status": "absent",
          "change": "resolved",
          "confidence": 0.9,
          "evidence": "Previously visible abnormality is no longer apparent."
        }
      ],
      "overall_change": "improved",
      "summary": "Pneumothorax appears resolved."
    }
    """

    result = (
        parse_temporal_vlm_response(
            raw_output=raw,

            pair_id="42_0_1",

            prior_study_id="prior",

            current_study_id="current",
        )
    )

    assert (
        result.pair_id
        == "42_0_1"
    )

    assert (
        len(
            result.findings
        )
        == 1
    )

    assert (
        result.findings[
            0
        ].change
        == "resolved"
    )

    assert (
        result.overall_change
        == "improved"
    )
def test_repairs_missing_comma():
    raw = """
    {
      "findings": [
        {
          "finding": "pneumothorax",
          "prior_status": "present",
          "current_status": "absent",
          "change": "resolved",
          "confidence": 0.9,
          "evidence": "Finding no longer visible"
        }
      ]
      "overall_change": "improved",
      "summary": "Pneumothorax resolved."
    }
    """

    result = parse_temporal_vlm_response(
        raw_output=raw,
        pair_id="test-pair",
        prior_study_id="prior",
        current_study_id="current",
    )

    assert (
        result.overall_change
        == "improved"
    )

    assert (
        result.findings[
            0
        ].change
        == "resolved"
    )


def test_schema_validation_still_rejects_bad_change():
    raw = """
    {
      "findings": [
        {
          "finding": "pneumothorax",
          "prior_status": "present",
          "current_status": "absent",
          "change": "much_better",
          "confidence": 0.9,
          "evidence": "Finding no longer visible"
        }
      ],
      "overall_change": "improved",
      "summary": "Resolved."
    }
    """

    import pytest

    with pytest.raises(
        ValueError,
        match="schema validation",
    ):
        parse_temporal_vlm_response(
            raw_output=raw,
            pair_id="test-pair",
            prior_study_id="prior",
            current_study_id="current",
        )