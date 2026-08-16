import pytest

from medchange.safety.config import (
    SafetyPolicyConfig,
)


def test_default_policy():
    config = SafetyPolicyConfig()

    assert (
        config.policy
        == "change_sensitive"
    )

    assert (
        config.threshold
        == 0.80
    )


def test_strict_default_threshold():
    config = SafetyPolicyConfig(
        policy="strict"
    )

    assert (
        config.threshold
        == 0.60
    )


def test_custom_threshold():
    config = SafetyPolicyConfig(
        policy="change_sensitive",
        threshold=0.75,
    )

    assert (
        config.threshold
        == 0.75
    )


def test_invalid_policy():
    with pytest.raises(
        ValueError,
        match="Unsupported safety policy",
    ):
        SafetyPolicyConfig(
            policy="invalid"
        )


def test_invalid_threshold():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        SafetyPolicyConfig(
            threshold=1.5
        )