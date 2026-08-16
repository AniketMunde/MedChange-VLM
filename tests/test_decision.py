from medchange.reasoning.decision import (
    derive_overall_change,
    resolve_temporal_decision,
)
from medchange.reasoning.evidence import (
    build_model_evidence,
)


def test_high_agreement():
    bio = build_model_evidence(
        "resolved",
        0.80,
    )

    qwen = build_model_evidence(
        "resolved",
        0.90,
    )

    decision = (
        resolve_temporal_decision(
            bio,
            qwen,
        )
    )

    assert (
        decision.final_state
        == "resolved"
    )

    assert (
        decision.agreement
        == "high"
    )

    assert (
        decision.uncertainty
        == "low"
    )

    assert not (
        decision.requires_review
    )


def test_conflict_abstains():
    bio = build_model_evidence(
        "absent",
        0.75,
    )

    qwen = build_model_evidence(
        "new",
        0.82,
    )

    decision = (
        resolve_temporal_decision(
            bio,
            qwen,
        )
    )

    assert (
        decision.final_state
        == "uncertain"
    )

    assert (
        decision.agreement
        == "conflict"
    )

    assert (
        decision.uncertainty
        == "high"
    )

    assert (
        decision.requires_review
    )


def test_both_uncertain():
    bio = build_model_evidence(
        "uncertain"
    )

    qwen = build_model_evidence(
        "uncertain"
    )

    decision = (
        resolve_temporal_decision(
            bio,
            qwen,
        )
    )

    assert (
        decision.final_state
        == "uncertain"
    )

    assert (
        decision.agreement
        == "unavailable"
    )

    assert (
        decision.requires_review
    )


def test_qwen_only_available():
    bio = build_model_evidence(
        "uncertain"
    )

    qwen = build_model_evidence(
        "new",
        0.80,
    )

    decision = (
        resolve_temporal_decision(
            bio,
            qwen,
        )
    )

    assert (
        decision.final_state
        == "new"
    )

    assert (
        decision.agreement
        == "partial"
    )

    assert (
        decision.uncertainty
        == "moderate"
    )

    assert (
        decision.requires_review
    )


def test_biomedclip_only_available():
    bio = build_model_evidence(
        "persistent",
        0.75,
    )

    qwen = build_model_evidence(
        "uncertain"
    )

    decision = (
        resolve_temporal_decision(
            bio,
            qwen,
        )
    )

    assert (
        decision.final_state
        == "persistent"
    )

    assert (
        decision.agreement
        == "partial"
    )

    assert (
        decision.requires_review
    )


def test_overall_worsened():
    assert (
        derive_overall_change(
            [
                "absent",
                "new",
                "persistent",
            ]
        )
        == "worsened"
    )


def test_overall_improved():
    assert (
        derive_overall_change(
            [
                "resolved",
                "absent",
            ]
        )
        == "improved"
    )


def test_overall_mixed():
    assert (
        derive_overall_change(
            [
                "new",
                "resolved",
            ]
        )
        == "mixed"
    )


def test_overall_stable():
    assert (
        derive_overall_change(
            [
                "persistent",
                "absent",
            ]
        )
        == "stable"
    )


def test_overall_uncertain_if_any_finding_uncertain():
    assert (
        derive_overall_change(
            [
                "absent",
                "uncertain",
                "persistent",
            ]
        )
        == "uncertain"
    )