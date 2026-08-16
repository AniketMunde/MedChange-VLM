from __future__ import annotations

import json
from dataclasses import dataclass

from json_repair import repair_json
from pydantic import ValidationError

from medchange.inference.structured_output import (
    extract_json,
)
from medchange.schemas_temporal import (
    TemporalPrediction,
)


@dataclass(frozen=True)
class TemporalParseResult:
    prediction: TemporalPrediction

    json_repaired: bool


def _repair_temporal_json(
    raw_output: str,
) -> dict:
    repaired = repair_json(
        raw_output
    )

    if not repaired:
        raise ValueError(
            "JSON repair returned empty output."
        )

    try:
        payload = json.loads(
            repaired
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Temporal output remained malformed "
            "after JSON repair."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Temporal model output must "
            "resolve to a JSON object."
        )

    return payload


def parse_temporal_vlm_response_with_metadata(
    raw_output: str,
    pair_id: str,
    prior_study_id: str,
    current_study_id: str,
) -> TemporalParseResult:
    """
    Parse Qwen temporal output and report whether
    syntax repair was required.
    """

    repaired = False

    try:
        payload = extract_json(
            raw_output
        )

    except ValueError as exc:
        if (
            "malformed JSON"
            not in str(exc)
            and "No JSON object found"
            not in str(exc)
        ):
            raise

        repaired = True

        payload = _repair_temporal_json(
            raw_output
        )

    payload[
        "pair_id"
    ] = pair_id

    payload[
        "prior_study_id"
    ] = prior_study_id

    payload[
        "current_study_id"
    ] = current_study_id

    try:
        prediction = (
            TemporalPrediction
            .model_validate(
                payload
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "Temporal JSON was syntactically valid "
            "but failed schema validation."
        ) from exc

    return TemporalParseResult(
        prediction=prediction,
        json_repaired=repaired,
    )


def parse_temporal_vlm_response(
    raw_output: str,
    pair_id: str,
    prior_study_id: str,
    current_study_id: str,
) -> TemporalPrediction:
    """
    Backwards-compatible parser.
    """

    result = (
        parse_temporal_vlm_response_with_metadata(
            raw_output=raw_output,
            pair_id=pair_id,
            prior_study_id=prior_study_id,
            current_study_id=current_study_id,
        )
    )

    return result.prediction