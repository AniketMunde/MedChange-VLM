from __future__ import annotations

from dataclasses import dataclass


VALID_TEMPORAL_STATES = {
    "absent",
    "new",
    "persistent",
    "resolved",
    "uncertain",
}


@dataclass(frozen=True)
class ModelTemporalEvidence:
    state: str

    confidence: float | None = None

    explanation: str | None = None


def normalize_temporal_state(
    state: str | None,
) -> str:
    if state is None:
        return "uncertain"

    normalized = (
        str(state)
        .strip()
        .lower()
    )

    if normalized not in (
        VALID_TEMPORAL_STATES
    ):
        return "uncertain"

    return normalized


def normalize_confidence(
    confidence: float | None,
) -> float | None:
    if confidence is None:
        return None

    value = float(
        confidence
    )

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def build_model_evidence(
    state: str | None,
    confidence: float | None = None,
    explanation: str | None = None,
) -> ModelTemporalEvidence:

    return ModelTemporalEvidence(
        state=normalize_temporal_state(
            state
        ),
        confidence=normalize_confidence(
            confidence
        ),
        explanation=explanation,
    )