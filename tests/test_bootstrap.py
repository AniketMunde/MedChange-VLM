import pytest
from sklearn.metrics import (
    roc_auc_score,
)

from medchange.evaluation.bootstrap import (
    bootstrap_metric,
)


def test_bootstrap_perfect_classifier():
    result = bootstrap_metric(
        y_true=[
            0,
            0,
            1,
            1,
            0,
            1,
        ],

        y_score=[
            0.1,
            0.2,
            0.8,
            0.9,
            0.3,
            0.7,
        ],

        metric_fn=roc_auc_score,

        n_bootstrap=100,

        seed=42,
    )

    assert (
        result.estimate
        == pytest.approx(1.0)
    )

    assert (
        result.lower
        <= result.estimate
        <= result.upper
    )


def test_bootstrap_single_class_rejected():
    with pytest.raises(
        ValueError,
        match="both positive",
    ):
        bootstrap_metric(
            y_true=[
                0,
                0,
                0,
            ],

            y_score=[
                0.1,
                0.2,
                0.3,
            ],

            metric_fn=roc_auc_score,
        )