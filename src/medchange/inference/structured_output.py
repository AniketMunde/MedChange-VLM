from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from medchange.schemas import (
    FindingLocation,
    MedicalFinding,
    MedChangePrediction,
    TemporalStatus,
    UncertaintyResult,
)


class VLMFinding(BaseModel):
    name: str

    present: bool = True

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    anatomy: str | None = None


class VLMResponse(BaseModel):
    findings: list[VLMFinding] = (
        Field(default_factory=list)
    )

    impression: str | None = None

    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    requires_review: bool = True


def extract_json(
    text: str,
) -> dict[str, Any]:
    """
    Extract a JSON object from VLM output.

    Supports:
    - plain JSON
    - markdown-fenced JSON
    - text surrounding JSON
    - leading/trailing whitespace

    Distinguishes between:
    - no JSON object present
    - malformed/incomplete JSON
    """

    cleaned = text.strip()

    if not cleaned:
        raise ValueError(
            "Model output is empty."
        )

    # Remove optional opening markdown fence.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove optional closing markdown fence.
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # First attempt: entire response is valid JSON.
    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError(
                "Top-level model output must be a JSON object."
            )

        return parsed

    except json.JSONDecodeError:
        pass

    # Look for evidence of a JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    # No opening brace means there is no JSON object at all.
    if start == -1:
        raise ValueError(
            "No JSON object found in model output."
        )

    # Opening brace exists but closing brace does not.
    # This is malformed/incomplete JSON.
    if end == -1:
        raise ValueError(
            "Model produced malformed JSON."
        )

    if end <= start:
        raise ValueError(
            "Model produced malformed JSON."
        )

    candidate = cleaned[start:end + 1]

    try:
        parsed = json.loads(candidate)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model produced malformed JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Top-level model output must be a JSON object."
        )

    return parsed


def parse_vlm_response(
    raw_output: str,
    study_id: str,
) -> MedChangePrediction:
    """
    Convert raw Qwen output into canonical MedChangePrediction.
    """

    data = extract_json(
        raw_output
    )

    response = VLMResponse.model_validate(
        data
    )

    findings: list[
        MedicalFinding
    ] = []

    for item in response.findings:

        location = None

        if item.anatomy:
            location = FindingLocation(
                anatomy=item.anatomy,
            )

        finding = MedicalFinding(
            name=item.name,
            present=item.present,
            confidence=item.confidence,

            temporal_status=(
                TemporalStatus.UNCERTAIN
            ),

            location=location,
        )

        findings.append(
            finding
        )

    return MedChangePrediction(
        study_id=study_id,

        findings=findings,

        impression=(
            response.impression
        ),

        uncertainty=(
            UncertaintyResult(
                overall_confidence=(
                    response.overall_confidence
                ),
                requires_review=True,
            )
        ),
    )