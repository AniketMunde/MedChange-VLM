from __future__ import annotations

from medchange.reasoning.decision import (
    derive_overall_change,
    resolve_temporal_decision,
)
from medchange.reasoning.evidence import (
    build_model_evidence,
)
from medchange.reasoning.temporal_result import (
    FindingEvidence,
    UnifiedTemporalResult,
)
from medchange.safety.config import (
    SafetyPolicyConfig,
)
from medchange.safety.policies import (
    apply_tuned_policy,
)

TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def build_unified_temporal_result(
    pair_id: str,
    prior_study_id: str,
    current_study_id: str,
    biomedclip_predictions: dict,
    qwen_predictions: dict,
    qwen_explanations: dict | None = None,
    safety_config: SafetyPolicyConfig | None = None,
) -> UnifiedTemporalResult:
    """
    Combine temporal evidence into one MedChange result.

    Expected dictionaries:

    biomedclip_predictions:
    {
        "pneumothorax": {
            "state": "resolved",
            "confidence": 0.81
        }
    }

    qwen_predictions:
    {
        "pneumothorax": {
            "state": "resolved",
            "confidence": 0.76
        }
    }
    """
    if safety_config is None:
        safety_config = (
            SafetyPolicyConfig()
        )

    qwen_explanations = (
        qwen_explanations
        or {}
    )

    findings = []

    for finding in (
        TARGET_FINDINGS
    ):

        bio_payload = (
            biomedclip_predictions
            .get(
                finding,
                {},
            )
        )

        qwen_payload = (
            qwen_predictions
            .get(
                finding,
                {},
            )
        )

        bio = (
            build_model_evidence(
                state=(
                    bio_payload
                    .get(
                        "state"
                    )
                ),
                confidence=(
                    bio_payload
                    .get(
                        "confidence"
                    )
                ),
            )
        )

        qwen = (
            build_model_evidence(
                state=(
                    qwen_payload
                    .get(
                        "state"
                    )
                ),
                confidence=(
                    qwen_payload
                    .get(
                        "confidence"
                    )
                ),
                explanation=(
                    qwen_explanations
                    .get(
                        finding
                    )
                ),
            )
        )

        decision = (
            resolve_temporal_decision(
                biomedclip=bio,
                qwen=qwen,
            )
        )
        safety = apply_tuned_policy(
            policy=(
                safety_config.policy
            ),
            biomedclip_state=(
                bio.state
            ),
            qwen_state=(
                qwen.state
            ),
            biomedclip_confidence=(
                bio.confidence
            ),
            threshold=(
                safety_config.threshold
            ),
        )

        findings.append(
            FindingEvidence(
                finding=finding,

                final_state=(
                    safety.final_state
                ),

                biomedclip_state=(
                    bio.state
                ),

                qwen_state=(
                    qwen.state
                ),

                biomedclip_confidence=(
                    bio.confidence
                ),

                qwen_confidence=(
                    qwen.confidence
                ),

                agreement=(
                    decision.agreement
                ),

                uncertainty=(
                    safety.uncertainty
                ),

                evidence=(
                    qwen.explanation
                ),
                decision_reason=(
                    f"{decision.reason} "
                    f"Configured safety policy "
                    f"'{safety_config.policy}' "
                    f"(threshold="
                    f"{safety_config.threshold:.2f}): "
                    f"{safety.reason}"
                ),
                requires_review=(
                    safety
                    .requires_review
                ),
            )
        )

    final_states = [
        finding.final_state
        for finding
        in findings
    ]

    overall_change = (
        derive_overall_change(
            final_states
        )
    )

    review_required = any(
        finding.requires_review
        for finding in findings
    )

    if review_required:
        overall_uncertainty = "high"

    elif any(
        finding.uncertainty
        == "moderate"
        for finding
        in findings
    ):
        overall_uncertainty = (
            "moderate"
        )

    else:
        overall_uncertainty = "low"

    changed = [
        finding
        for finding
        in findings
        if finding.final_state
        in {
            "new",
            "resolved",
        }
    ]

    if changed:
        summary_parts = [
            (
                f"{finding.finding}: "
                f"{finding.final_state}"
            )
            for finding
            in changed
        ]

        summary = (
            "; ".join(
                summary_parts
            )
        )

    else:
        summary = (
            "No new or resolved target "
            "findings detected."
        )

    return UnifiedTemporalResult(
        pair_id=pair_id,

        prior_study_id=(
            prior_study_id
        ),

        current_study_id=(
            current_study_id
        ),

        findings=findings,

        overall_change=(
            overall_change
        ),

        uncertainty=(
            overall_uncertainty
        ),

        requires_review=(
            review_required
        ),

        summary=summary,
    )