from __future__ import annotations

import time

import pandas as pd
import torch

from medchange.data.nih import (
    TARGET_FINDINGS,
    iter_nih_examples,
)
from medchange.models.vision import (
    BiomedCLIP,
)


class NIHEvaluationRunner:
    """
    Stream NIH ChestXray14 directly from Hugging Face
    and run BiomedCLIP zero-shot inference.
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
            else TARGET_FINDINGS
        )

    def run(
        self,
        split: str,
        max_samples: int,
        view: str = "ALL",
        shuffle: bool = True,
        seed: int = 42,
        shuffle_buffer: int = 2000,
    ) -> pd.DataFrame:
        """
        Evaluate a streamed subset and return prediction rows.
        """

        self.model.warm_text_cache(
            self.findings
        )

        rows: list[dict] = []

        total_start = (
            time.perf_counter()
        )

        examples = iter_nih_examples(
            split=split,
            max_samples=max_samples,
            view=view,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )

        for index, example in enumerate(
            examples,
            start=1,
        ):
            start = (
                time.perf_counter()
            )

            scores = (
                self.model.score_findings(
                    image_input=example.image,
                    findings=self.findings,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            row = {
                "sample_id": (
                    f"{split}-{index:06d}"
                ),

                "patient_id": (
                    example.patient_id
                ),

                "view_position": (
                    example.view_position
                ),

                "patient_age": (
                    example.patient_age
                ),

                "patient_gender": (
                    example.patient_gender
                ),

                "raw_labels": "|".join(
                    example.raw_labels
                ),

                "inference_seconds": (
                    elapsed
                ),
            }

            for finding in self.findings:
                row[
                    f"{finding}_label"
                ] = example.labels[
                    finding
                ]

                row[
                    f"{finding}_score"
                ] = scores[
                    finding
                ]

            rows.append(
                row
            )

            if (
                index == 1
                or index % 25 == 0
                or index == max_samples
            ):
                print(
                    f"[{index}/{max_samples}] "
                    f"patient={example.patient_id} "
                    f"view={example.view_position} "
                    f"time={elapsed:.3f}s"
                )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        print()
        print(
            f"Processed {len(rows)} images "
            f"in {total_elapsed:.2f} seconds."
        )

        if rows:
            print(
                "Mean end-to-end time/image: "
                f"{total_elapsed / len(rows):.3f}s"
            )

        return pd.DataFrame(
            rows
        )