import numpy as np

from medchange.evaluation.selective_metrics import (
    compute_selective_metrics,
)


def test_selective_metrics_without_abstention():
    y_true = np.asarray(
        [
            "absent",
            "new",
            "resolved",
            "absent",
        ]
    )

    y_pred = np.asarray(
        [
            "absent",
            "new",
            "absent",
            "absent",
        ]
    )

    metrics = (
        compute_selective_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    assert (
        metrics[
            "coverage"
        ]
        == 1.0
    )

    assert (
        metrics[
            "abstention_rate"
        ]
        == 0.0
    )

    assert (
        metrics[
            "selective_accuracy"
        ]
        == 0.75
    )


def test_selective_metrics_with_abstention():
    y_true = np.asarray(
        [
            "absent",
            "new",
            "resolved",
            "absent",
        ]
    )

    y_pred = np.asarray(
        [
            "absent",
            "uncertain",
            "resolved",
            "uncertain",
        ]
    )

    review = np.asarray(
        [
            False,
            True,
            False,
            True,
        ]
    )

    metrics = (
        compute_selective_metrics(
            y_true=y_true,
            y_pred=y_pred,
            requires_review=review,
        )
    )

    assert (
        metrics[
            "coverage"
        ]
        == 0.5
    )

    assert (
        metrics[
            "abstention_rate"
        ]
        == 0.5
    )

    assert (
        metrics[
            "selective_accuracy"
        ]
        == 1.0
    )

    assert (
        metrics[
            "error_rate_on_covered"
        ]
        == 0.0
    )

    assert (
        metrics[
            "review_rate"
        ]
        == 0.5
    )


def test_all_abstained():
    y_true = np.asarray(
        [
            "new",
            "resolved",
        ]
    )

    y_pred = np.asarray(
        [
            "uncertain",
            "uncertain",
        ]
    )

    metrics = (
        compute_selective_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    assert (
        metrics[
            "coverage"
        ]
        == 0.0
    )

    assert (
        metrics[
            "selective_accuracy"
        ]
        is None
    )