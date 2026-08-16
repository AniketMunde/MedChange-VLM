from medchange.safety.policies import (
    apply_tuned_policy,
)


def test_strict_conflict_abstains():
    decision = (
        apply_tuned_policy(
            policy="strict",
            biomedclip_state="new",
            qwen_state="absent",
            biomedclip_confidence=0.9,
            threshold=0.6,
        )
    )

    assert (
        decision.final_state
        == "uncertain"
    )

    assert (
        decision.abstained
        is True
    )


def test_confidence_margin_preserves_high_confidence():
    decision = (
        apply_tuned_policy(
            policy="confidence_margin",
            biomedclip_state="new",
            qwen_state="absent",
            biomedclip_confidence=0.9,
            threshold=0.6,
        )
    )

    assert (
        decision.final_state
        == "new"
    )

    assert (
        decision.abstained
        is False
    )

    assert (
        decision.requires_review
        is True
    )


def test_confidence_margin_abstains_low_confidence():
    decision = (
        apply_tuned_policy(
            policy="confidence_margin",
            biomedclip_state="new",
            qwen_state="absent",
            biomedclip_confidence=0.4,
            threshold=0.6,
        )
    )

    assert (
        decision.final_state
        == "uncertain"
    )

    assert (
        decision.abstained
        is True
    )


def test_change_sensitive_preserves_change():
    decision = (
        apply_tuned_policy(
            policy="change_sensitive",
            biomedclip_state="resolved",
            qwen_state="absent",
            biomedclip_confidence=0.4,
            threshold=0.6,
        )
    )

    assert (
        decision.final_state
        == "resolved"
    )

    assert (
        decision.requires_review
        is True
    )

    assert (
        decision.abstained
        is False
    )


def test_low_confidence_only_ignores_conflict_for_state():
    decision = (
        apply_tuned_policy(
            policy="low_confidence_only",
            biomedclip_state="new",
            qwen_state="absent",
            biomedclip_confidence=0.9,
            threshold=0.6,
        )
    )

    assert (
        decision.final_state
        == "new"
    )

    assert (
        decision.abstained
        is False
    )

    assert (
        decision.requires_review
        is True
    )