from __future__ import annotations

from dataclasses import dataclass


CHANGE_STATES = {
    "new",
    "resolved",
    "persistent",
}


@dataclass(frozen=True)
class TunedSafetyDecision:
    final_state: str

    uncertainty: str

    requires_review: bool

    abstained: bool

    reason: str


def apply_tuned_policy(
    *,
    policy: str,
    biomedclip_state: str,
    qwen_state: str,
    biomedclip_confidence: float | None,
    threshold: float,
) -> TunedSafetyDecision:
    """
    Evaluate alternative abstention policies.

    These policies are for offline evaluation first.
    They should not replace the deployed policy until
    M5.5.2 has been analyzed.
    """

    bio_state = str(
        biomedclip_state
    )

    qwen_state = str(
        qwen_state
    )

    confidence = (
        None
        if biomedclip_confidence
        is None
        else float(
            biomedclip_confidence
        )
    )

    conflict = (
        bio_state
        != qwen_state
    )

    low_confidence = (
        confidence is None
        or confidence
        < threshold
    )

    # --------------------------------------------------
    # STRICT
    # --------------------------------------------------

    if policy == "strict":

        if conflict:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Strict policy abstained "
                    "because models disagree."
                ),
            )

        if low_confidence:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Strict policy abstained "
                    "because BiomedCLIP confidence "
                    "is below threshold."
                ),
            )

        return TunedSafetyDecision(
            final_state=bio_state,
            uncertainty="low",
            requires_review=False,
            abstained=False,
            reason=(
                "Models agree and confidence "
                "passes threshold."
            ),
        )

    # --------------------------------------------------
    # CONFIDENCE-MARGIN
    # --------------------------------------------------

    if policy == "confidence_margin":

        if (
            conflict
            and low_confidence
        ):
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Models disagree and "
                    "BiomedCLIP confidence "
                    "is below threshold."
                ),
            )

        if conflict:
            return TunedSafetyDecision(
                final_state=bio_state,
                uncertainty="moderate",
                requires_review=True,
                abstained=False,
                reason=(
                    "Models disagree, but "
                    "BiomedCLIP confidence "
                    "passes threshold."
                ),
            )

        if low_confidence:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Models agree but "
                    "BiomedCLIP confidence "
                    "is below threshold."
                ),
            )

        return TunedSafetyDecision(
            final_state=bio_state,
            uncertainty="low",
            requires_review=False,
            abstained=False,
            reason=(
                "Prediction passed "
                "confidence-margin policy."
            ),
        )

    # --------------------------------------------------
    # CHANGE-SENSITIVE
    # --------------------------------------------------

    if policy == "change_sensitive":

        if (
            bio_state
            in CHANGE_STATES
        ):
            if conflict:
                return TunedSafetyDecision(
                    final_state=bio_state,
                    uncertainty="moderate",
                    requires_review=True,
                    abstained=False,
                    reason=(
                        "Change-sensitive policy "
                        "preserved BiomedCLIP "
                        "change-state prediction "
                        "despite Qwen disagreement."
                    ),
                )

            if low_confidence:
                return TunedSafetyDecision(
                    final_state=bio_state,
                    uncertainty="moderate",
                    requires_review=True,
                    abstained=False,
                    reason=(
                        "Change-sensitive policy "
                        "preserved a low-confidence "
                        "change-state prediction."
                    ),
                )

            return TunedSafetyDecision(
                final_state=bio_state,
                uncertainty="low",
                requires_review=False,
                abstained=False,
                reason=(
                    "BiomedCLIP change-state "
                    "prediction passed policy."
                ),
            )

        # For ABSENT predictions remain conservative.

        if conflict:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Absent prediction conflicts "
                    "with Qwen; policy abstained."
                ),
            )

        if low_confidence:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "Low-confidence absent "
                    "prediction was abstained."
                ),
            )

        return TunedSafetyDecision(
            final_state=bio_state,
            uncertainty="low",
            requires_review=False,
            abstained=False,
            reason=(
                "Absent prediction passed "
                "change-sensitive policy."
            ),
        )

    # --------------------------------------------------
    # LOW-CONFIDENCE ONLY
    # --------------------------------------------------

    if policy == "low_confidence_only":

        if low_confidence:
            return TunedSafetyDecision(
                final_state="uncertain",
                uncertainty="high",
                requires_review=True,
                abstained=True,
                reason=(
                    "BiomedCLIP confidence "
                    "is below threshold."
                ),
            )

        return TunedSafetyDecision(
            final_state=bio_state,
            uncertainty=(
                "moderate"
                if conflict
                else "low"
            ),
            requires_review=(
                conflict
            ),
            abstained=False,
            reason=(
                "BiomedCLIP confidence "
                "passes threshold."
            ),
        )

    raise ValueError(
        f"Unknown policy: {policy}"
    )