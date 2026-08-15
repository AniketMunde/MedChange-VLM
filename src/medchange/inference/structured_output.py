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

    Handles occasional markdown fences or extra model text.
    """

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in model output."
        )

    candidate = cleaned[
        start:end + 1
    ]

    try:
        return json.loads(candidate)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model produced malformed JSON."
        ) from exc


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