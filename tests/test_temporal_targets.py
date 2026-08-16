import pytest

from medchange.data.nih.temporal_targets import (
    TemporalTarget,
    derive_all_temporal_targets,
    derive_finding_temporal_target,
)


def test_new_finding():
    state = derive_finding_temporal_target(
        finding="Effusion",
        prior_labels=("No Finding",),
        current_labels=("Effusion",),
    )

    assert state == TemporalTarget.NEW


def test_resolved_finding():
    state = derive_finding_temporal_target(
        finding="Pneumothorax",
        prior_labels=("Pneumothorax",),
        current_labels=("No Finding",),
    )

    assert (
        state
        == TemporalTarget.RESOLVED
    )


def test_persistent_finding():
    state = derive_finding_temporal_target(
        finding="Cardiomegaly",
        prior_labels=("Cardiomegaly",),
        current_labels=("Cardiomegaly",),
    )

    assert (
        state
        == TemporalTarget.PERSISTENT
    )


def test_absent_finding():
    state = derive_finding_temporal_target(
        finding="Edema",
        prior_labels=("No Finding",),
        current_labels=("No Finding",),
    )

    assert (
        state
        == TemporalTarget.ABSENT
    )


def test_all_targets():
    result = derive_all_temporal_targets(
        prior_labels=(
            "Cardiomegaly",
            "Effusion",
        ),
        current_labels=(
            "Cardiomegaly",
            "Atelectasis",
        ),
    )

    assert (
        result["atelectasis"]
        == TemporalTarget.NEW
    )

    assert (
        result["pleural_effusion"]
        == TemporalTarget.RESOLVED
    )

    assert (
        result["cardiomegaly"]
        == TemporalTarget.PERSISTENT
    )

    assert (
        result["pneumothorax"]
        == TemporalTarget.ABSENT
    )


def test_unknown_finding():
    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        derive_finding_temporal_target(
            finding="Unknown Disease",
            prior_labels=(),
            current_labels=(),
        )