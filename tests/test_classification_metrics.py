import pytest

from medchange.evaluation import (
    compute_binary_metrics,
)


def test_perfect_binary_classifier():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_score = [
        0.1,
        0.2,
        0.8,
        0.9,
    ]

    metrics = (
        compute_binary_metrics(
            y_true,
            y_score,
        )
    )

    assert (
        metrics.auroc
        == pytest.approx(1.0)
    )

    assert (
        metrics.auprc
        == pytest.approx(1.0)
    )

    assert (
        metrics.f1
        == pytest.approx(1.0)
    )

def test_uncertain_and_missing_labels_excluded():
    metrics = compute_binary_metrics(
        y_true=[
            1,
            0,
            -1,
            float("nan"),
            1,
        ],
        y_score=[
            0.9,
            0.1,
            0.8,
            0.4,
            0.7,
        ],
    )

    assert metrics.n_total == 5

    assert metrics.n_evaluated == 3

    assert metrics.n_excluded == 2

    assert metrics.n_positive == 2

    assert metrics.n_negative == 1

    assert metrics.auroc == pytest.approx(
        1.0
    )

def test_shape_mismatch():
    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        compute_binary_metrics(
            [0, 1],
            [0.2],
        )


def test_empty_metrics_input():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        compute_binary_metrics(
            [],
            [],
        )


def test_single_class_auroc_is_none():
    metrics = (
        compute_binary_metrics(
            [0, 0, 0],
            [
                0.1,
                0.2,
                0.3,
            ],
        )
    )

    assert metrics.auroc is None
