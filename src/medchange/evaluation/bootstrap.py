from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    n_valid_bootstrap: int


def bootstrap_metric(
    y_true,
    y_score,
    metric_fn: Callable[
        [np.ndarray, np.ndarray],
        float,
    ],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """
    Estimate a metric and percentile bootstrap confidence interval.

    Bootstrap samples that do not contain both classes are skipped.
    """

    true = np.asarray(
        y_true,
        dtype=int,
    )

    score = np.asarray(
        y_score,
        dtype=float,
    )

    if true.shape != score.shape:
        raise ValueError(
            "y_true and y_score must have identical shapes."
        )

    if true.size == 0:
        raise ValueError(
            "Bootstrap inputs cannot be empty."
        )

    if np.unique(true).size < 2:
        raise ValueError(
            "Bootstrap metric requires both positive "
            "and negative classes."
        )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    estimate = float(
        metric_fn(
            true,
            score,
        )
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_values: list[float] = []

    n_samples = true.size

    for _ in range(
        n_bootstrap
    ):
        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        sample_true = true[
            indices
        ]

        sample_score = score[
            indices
        ]

        if (
            np.unique(
                sample_true
            ).size
            < 2
        ):
            continue

        value = metric_fn(
            sample_true,
            sample_score,
        )

        bootstrap_values.append(
            float(value)
        )

    if not bootstrap_values:
        raise ValueError(
            "No valid bootstrap samples were generated."
        )

    values = np.asarray(
        bootstrap_values
    )

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100.0
        * alpha
        / 2.0
    )

    upper_percentile = (
        100.0
        * (
            1.0
            - alpha / 2.0
        )
    )

    lower = float(
        np.percentile(
            values,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            values,
            upper_percentile,
        )
    )

    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_valid_bootstrap=len(
            bootstrap_values
        ),
    )