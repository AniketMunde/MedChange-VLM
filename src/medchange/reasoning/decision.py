from __future__ import annotations

from dataclasses import dataclass

from medchange.reasoning.evidence import (
    ModelTemporalEvidence,
)


@dataclass(frozen=True)
class TemporalDecision:
    final_state: str

    agreement: str

    uncertainty: str

    requires_review: bool

    reason: str


def resolve_temporal_decision(
    biomedclip: ModelTemporalEvidence,
    qwen: ModelTemporalEvidence,
) -> TemporalDecision:
    """
    Resolve BiomedCLIP and Qwen temporal evidence.

    Safety policy:
    - exact agreement -> accept
    - only one usable model -> use it, but flag review
    - direct model disagreement -> abstain
    - both uncertain -> abstain

    We intentionally do not use raw confidence alone to
    override disagreement because Qwen confidence was
    observed to be poorly calibrated in M4.4.
    """

    bio_state = biomedclip.state
    qwen_state = qwen.state

    # --------------------------------------------------
    # Both uncertain
    # --------------------------------------------------

    if (
        bio_state == "uncertain"
        and qwen_state == "uncertain"
    ):
        return TemporalDecision(
            final_state="uncertain",
            agreement="unavailable",
            uncertainty="high",
            requires_review=True,
            reason=(
                "Both models were unable to provide "
                "a reliable temporal state."
            ),
        )

    # --------------------------------------------------
    # Exact agreement
    # --------------------------------------------------

    if (
        bio_state == qwen_state
        and bio_state != "uncertain"
    ):
        return TemporalDecision(
            final_state=bio_state,
            agreement="high",
            uncertainty="low",
            requires_review=False,
            reason=(
                "BiomedCLIP and Qwen agree on "
                "the temporal state."
            ),
        )

    # --------------------------------------------------
    # BiomedCLIP uncertain, Qwen available
    # --------------------------------------------------

    if (
        bio_state == "uncertain"
        and qwen_state != "uncertain"
    ):
        return TemporalDecision(
            final_state=qwen_state,
            agreement="partial",
            uncertainty="moderate",
            requires_review=True,
            reason=(
                "Only Qwen provided a usable "
                "temporal prediction."
            ),
        )

    # --------------------------------------------------
    # Qwen uncertain, BiomedCLIP available
    # --------------------------------------------------

    if (
        qwen_state == "uncertain"
        and bio_state != "uncertain"
    ):
        return TemporalDecision(
            final_state=bio_state,
            agreement="partial",
            uncertainty="moderate",
            requires_review=True,
            reason=(
                "Only BiomedCLIP provided a usable "
                "temporal prediction."
            ),
        )

    # --------------------------------------------------
    # Direct conflict
    # --------------------------------------------------

    return TemporalDecision(
        final_state="uncertain",
        agreement="conflict",
        uncertainty="high",
        requires_review=True,
        reason=(
            "BiomedCLIP and Qwen produced "
            "conflicting temporal states."
        ),
    )


def derive_overall_change(
    states: list[str],
) -> str:
    """
    Derive pair-level temporal status.

    If any finding remains uncertain after evidence
    resolution, the overall result is uncertain.
    """

    if not states:
        return "uncertain"

    if "uncertain" in states:
        return "uncertain"

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