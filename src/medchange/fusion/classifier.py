from __future__ import annotations

import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.preprocessing import (
    StandardScaler,
)


class FusionClassifier:
    """
    Standardized multinomial logistic regression.

    Used for both BiomedCLIP-only and fused features
    so comparisons stay fair.
    """

    def __init__(
        self,
        seed: int = 42,
    ) -> None:

        self.scaler = (
            StandardScaler()
        )

        self.model = (
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:

        scaled = (
            self.scaler
            .fit_transform(
                x
            )
        )

        self.model.fit(
            scaled,
            y,
        )

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        scaled = (
            self.scaler
            .transform(
                x
            )
        )

        return self.model.predict(
            scaled
        )