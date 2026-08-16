from medchange.data.nih.change_labels import (
    derive_temporal_label_change,
)


def test_new_finding():
    change = derive_temporal_label_change(
        prior_labels=(
            "No Finding",
        ),
        current_labels=(
            "Effusion",
        ),
    )

    assert change.new == (
        "Effusion",
    )

    assert change.resolved == ()

    assert change.persistent == ()

    assert change.has_change


def test_resolved_finding():
    change = derive_temporal_label_change(
        prior_labels=(
            "Effusion",
        ),
        current_labels=(
            "No Finding",
        ),
    )

    assert change.new == ()

    assert change.resolved == (
        "Effusion",
    )

    assert change.has_change


def test_persistent_finding():
    change = derive_temporal_label_change(
        prior_labels=(
            "Effusion",
        ),
        current_labels=(
            "Effusion",
        ),
    )

    assert change.new == ()

    assert change.resolved == ()

    assert change.persistent == (
        "Effusion",
    )

    assert not change.has_change


def test_mixed_change():
    change = derive_temporal_label_change(
        prior_labels=(
            "Cardiomegaly",
            "Effusion",
        ),
        current_labels=(
            "Cardiomegaly",
            "Atelectasis",
        ),
    )

    assert change.new == (
        "Atelectasis",
    )

    assert change.resolved == (
        "Effusion",
    )

    assert change.persistent == (
        "Cardiomegaly",
    )

    assert change.has_change