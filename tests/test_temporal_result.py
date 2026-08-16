from medchange.reasoning.temporal_result import (
    FindingEvidence,
    UnifiedTemporalResult,
)


def test_unified_temporal_result():
    finding = FindingEvidence(
        finding="pneumothorax",
        final_state="resolved",
        biomedclip_state="resolved",
        qwen_state="resolved",
        biomedclip_confidence=0.81,
        qwen_confidence=0.76,
        agreement="high",
        uncertainty="low",
        evidence=(
            "Both temporal models support resolution."
        ),
        requires_review=False,
    )

    result = UnifiedTemporalResult(
        pair_id="42_0_1",
        prior_study_id="prior",
        current_study_id="current",
        findings=[
            finding
        ],
        overall_change="improved",
        uncertainty="low",
        requires_review=False,
        summary=(
            "Pneumothorax appears resolved."
        ),
    )

    assert (
        result.findings[
            0
        ].final_state
        == "resolved"
    )

    assert (
        result.overall_change
        == "improved"
    )