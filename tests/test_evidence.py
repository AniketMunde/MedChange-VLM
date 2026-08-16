from medchange.reasoning.evidence import (
    build_model_evidence,
    normalize_confidence,
    normalize_temporal_state,
)


def test_unknown_state_becomes_uncertain():
    assert (
        normalize_temporal_state(
            "something_invalid"
        )
        == "uncertain"
    )


def test_confidence_is_clamped():
    assert (
        normalize_confidence(
            1.4
        )
        == 1.0
    )

    assert (
        normalize_confidence(
            -0.2
        )
        == 0.0
    )


def test_build_model_evidence():
    evidence = (
        build_model_evidence(
            state="new",
            confidence=0.8,
            explanation="New opacity.",
        )
    )

    assert (
        evidence.state
        == "new"
    )

    assert (
        evidence.confidence
        == 0.8
    )