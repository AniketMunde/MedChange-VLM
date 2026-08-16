from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MIN_BIOMEDCLIP_CONFIDENCE = 0.60
DEFAULT_MIN_QWEN_CONFIDENCE = 0.60


@dataclass(frozen=True)
class SafetyDecision:
    final_state: str
    uncertainty: str
    requires_review: bool
    abstained: bool
    reason: str


def apply_safety_policy(
    *,
    final_state: str,
    agreement: str,
    biomedclip_confidence: float | None,
    qwen_confidence: float | None,
    min_biomedclip_confidence: float = (
        DEFAULT_MIN_BIOMEDCLIP_CONFIDENCE
    ),
    min_qwen_confidence: float = (
        DEFAULT_MIN_QWEN_CONFIDENCE
    ),
) -> SafetyDecision:
    """
    Apply deterministic safety rules after model reconciliation.

    Principles:
    - conflicts always abstain
    - missing confidence is treated cautiously
    - low BiomedCLIP confidence can force abstention
    - low Qwen confidence alone does not override BiomedCLIP
    - uncertain final states remain abstentions
    """

    if final_state == "uncertain":
        return SafetyDecision(
            final_state="uncertain",
            uncertainty="high",
            requires_review=True,
            abstained=True,
            reason=(
                "Temporal state is already uncertain."
            ),
        )

    if agreement == "conflict":
        return SafetyDecision(
            final_state="uncertain",
            uncertainty="high",
            requires_review=True,
            abstained=True,
            reason=(
                "BiomedCLIP and Qwen disagree."
            ),
        )

    if biomedclip_confidence is None:
        return SafetyDecision(
            final_state="uncertain",
            uncertainty="high",
            requires_review=True,
            abstained=True,
            reason=(
                "BiomedCLIP confidence is unavailable."
            ),
        )

    if (
        biomedclip_confidence
        < min_biomedclip_confidence
    ):
        return SafetyDecision(
            final_state="uncertain",
            uncertainty="high",
            requires_review=True,
            abstained=True,
            reason=(
                "BiomedCLIP confidence is below "
                "the configured safety threshold."
            ),
        )

    if (
        agreement == "high"
        and qwen_confidence is not None
        and qwen_confidence
        < min_qwen_confidence
    ):
        return SafetyDecision(
            final_state=final_state,
            uncertainty="moderate",
            requires_review=True,
            abstained=False,
            reason=(
                "Models agree, but Qwen confidence "
                "is below threshold."
            ),
        )

    return SafetyDecision(
        final_state=final_state,
        uncertainty="low",
        requires_review=False,
        abstained=False,
        reason=(
            "Prediction passed the configured "
            "safety checks."
        ),
    )