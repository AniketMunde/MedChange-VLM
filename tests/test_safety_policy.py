from medchange.safety.policy import (
    apply_safety_policy,
)


def test_conflict_abstains():
    result = apply_safety_policy(
        final_state="new",
        agreement="conflict",
        biomedclip_confidence=0.9,
        qwen_confidence=0.9,
    )

    assert result.final_state == "uncertain"
    assert result.abstained is True
    assert result.requires_review is True


def test_low_biomedclip_confidence_abstains():
    result = apply_safety_policy(
        final_state="resolved",
        agreement="high",
        biomedclip_confidence=0.40,
        qwen_confidence=0.95,
    )

    assert result.final_state == "uncertain"
    assert result.abstained is True


def test_missing_biomedclip_confidence_abstains():
    result = apply_safety_policy(
        final_state="persistent",
        agreement="high",
        biomedclip_confidence=None,
        qwen_confidence=0.90,
    )

    assert result.final_state == "uncertain"
    assert result.requires_review is True


def test_low_qwen_confidence_marks_review():
    result = apply_safety_policy(
        final_state="absent",
        agreement="high",
        biomedclip_confidence=0.92,
        qwen_confidence=0.40,
    )

    assert result.final_state == "absent"
    assert result.abstained is False
    assert result.requires_review is True
    assert result.uncertainty == "moderate"


def test_high_agreement_passes():
    result = apply_safety_policy(
        final_state="new",
        agreement="high",
        biomedclip_confidence=0.88,
        qwen_confidence=0.82,
    )

    assert result.final_state == "new"
    assert result.abstained is False
    assert result.requires_review is False