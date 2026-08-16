from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from medchange.evaluation.constants import (
    EVALUATION_FINDINGS,
)
from medchange.models.vision import (
    BiomedCLIP,
)


class BiomedCLIPEvaluationRunner:
    """
    Run BiomedCLIP inference over an evaluation manifest.
    """

    def __init__(
        self,
        model: BiomedCLIP | None = None,
        findings: list[str] | None = None,
    ) -> None:

        self.model = (
            model
            if model is not None
            else BiomedCLIP()
        )

        self.findings = (
            findings
            if findings is not None
            else EVALUATION_FINDINGS
        )

    def run(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate pathology scores for every image.
        """

        self.model.warm_text_cache(
            self.findings
        )

        rows: list[dict] = []

        total = len(
            dataframe
        )

        for index, row in dataframe.iterrows():

            image_path = str(
                row["image_path"]
            )

            study_id = str(
                row["study_id"]
            )

            start_time = (
                time.perf_counter()
            )

            scores = (
                self.model.score_findings(
                    image_path=image_path,
                    findings=self.findings,
                )
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            result = {
                "image_path": (
                    image_path
                ),
                "study_id": (
                    study_id
                ),
                "inference_seconds": (
                    elapsed
                ),
            }

            for finding in self.findings:
                result[
                    f"{finding}_score"
                ] = scores[
                    finding
                ]

                label_column = (
                    f"{finding}_label"
                )

                if (
                    label_column
                    in dataframe.columns
                ):
                    result[
                        label_column
                    ] = row[
                        label_column
                    ]

            rows.append(
                result
            )

            print(
                f"[{index + 1}/{total}] "
                f"{study_id} "
                f"({elapsed:.3f}s)"
            )

        return pd.DataFrame(
            rows
        )