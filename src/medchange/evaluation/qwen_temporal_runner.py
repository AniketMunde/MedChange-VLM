
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)
from medchange.evaluation.qwen_temporal_metrics import (
    TARGET_FINDINGS,
    compute_qwen_temporal_metrics,
    prediction_to_confidence_map,
    prediction_to_state_map,
)
from medchange.inference.temporal_pipeline import (
    TemporalQwenPipeline,
)


class QwenTemporalBenchmarkRunner:
    def __init__(
        self,
        dataset_root: str | Path,
    ) -> None:

        self.resolver = NIHImageResolver(
            dataset_root
        )

        self.resolver.build_index()

        print(
            f"Indexed "
            f"{self.resolver.num_images:,} "
            "NIH images."
        )

        self.pipeline = (
            TemporalQwenPipeline()
        )

    def run(
        self,
        dataframe: pd.DataFrame,
        output_dir: str | Path,
    ) -> dict[str, Any]:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        finding_rows = []

        pair_rows = []

        failures = []

        total = len(
            dataframe
        )

        for position, (_, row) in enumerate(
            dataframe.iterrows(),
            start=1,
        ):
            pair_id = str(
                row[
                    "pair_id"
                ]
            )

            prior_index = str(
                row[
                    "prior_image_index"
                ]
            )

            current_index = str(
                row[
                    "current_image_index"
                ]
            )

            print()
            print(
                "=" * 90
            )

            print(
                f"[{position}/{total}] "
                f"pair={pair_id}"
            )

            print(
                "=" * 90
            )

            try:
                prior_path = (
                    self.resolver.resolve(
                        prior_index
                    )
                )

                current_path = (
                    self.resolver.resolve(
                        current_index
                    )
                )

                result = (
                    self.pipeline
                    .analyze_pair_detailed(
                        prior_image_path=(
                            prior_path
                        ),

                        current_image_path=(
                            current_path
                        ),

                        pair_id=pair_id,

                        prior_study_id=(
                            Path(
                                prior_index
                            ).stem
                        ),

                        current_study_id=(
                            Path(
                                current_index
                            ).stem
                        ),
                    )
                )

                prediction_map = (
                    prediction_to_state_map(
                        result.prediction
                    )
                )

                confidence_map = (
                    prediction_to_confidence_map(
                        result.prediction
                    )
                )

                pair_correct = True

                pair_finding_correct = 0

                for finding in (
                    TARGET_FINDINGS
                ):
                    ground_truth = str(
                        row[
                            f"{finding}_temporal"
                        ]
                    )

                    predicted = (
                        prediction_map.get(
                            finding,
                            "uncertain",
                        )
                    )

                    confidence = (
                        confidence_map.get(
                            finding,
                            0.0,
                        )
                    )

                    correct = (
                        predicted
                        == ground_truth
                    )

                    if correct:
                        pair_finding_correct += 1

                    else:
                        pair_correct = False

                    finding_rows.append(
                        {
                            "pair_id": (
                                pair_id
                            ),

                            "benchmark_category": (
                                row.get(
                                    "benchmark_category",
                                    "unknown",
                                )
                            ),

                            "finding": (
                                finding
                            ),

                            "ground_truth": (
                                ground_truth
                            ),

                            "prediction": (
                                predicted
                            ),

                            "confidence": (
                                confidence
                            ),

                            "correct": (
                                correct
                            ),
                        }
                    )

                pair_rows.append(
                    {
                        "pair_id": pair_id,

                        "benchmark_category": (
                            row.get(
                                "benchmark_category",
                                "unknown",
                            )
                        ),

                        "exact_match": (
                            pair_correct
                        ),

                        "num_correct_findings": (
                            pair_finding_correct
                        ),

                        "finding_accuracy": (
                            pair_finding_correct
                            / len(
                                TARGET_FINDINGS
                            )
                        ),

                        "json_repaired": (
                            result.json_repaired
                        ),

                        "elapsed_seconds": (
                            result.metrics
                            .elapsed_seconds
                        ),

                        "gpu_peak_gb": (
                            result.metrics
                            .gpu_peak_allocated_gb
                        ),
                    }
                )

                print(
                    f"Correct findings: "
                    f"{pair_finding_correct}/7"
                )

                print(
                    f"JSON repaired   : "
                    f"{result.json_repaired}"
                )

                print(
                    f"Elapsed         : "
                    f"{result.metrics.elapsed_seconds:.2f}s"
                )

            except Exception as exc:
                failures.append(
                    {
                        "pair_id": pair_id,
                        "error": str(
                            exc
                        ),
                    }
                )

                print(
                    f"FAILED: {exc}"
                )

        findings_df = pd.DataFrame(
            finding_rows
        )

        pairs_df = pd.DataFrame(
            pair_rows
        )

        failures_df = pd.DataFrame(
            failures
        )

        findings_df.to_csv(
            output_dir
            / "finding_predictions.csv",
            index=False,
        )

        pairs_df.to_csv(
            output_dir
            / "pair_results.csv",
            index=False,
        )

        failures_df.to_csv(
            output_dir
            / "failures.csv",
            index=False,
        )

        metrics = (
            compute_qwen_temporal_metrics(
                findings_df
            )
        )

        successful_pairs = len(
            pairs_df
        )

        metrics[
            "requested_pairs"
        ] = total

        metrics[
            "successful_pairs"
        ] = successful_pairs

        metrics[
            "failed_pairs"
        ] = len(
            failures_df
        )

        metrics[
            "parse_success_rate"
        ] = (
            successful_pairs
            / total
            if total > 0
            else 0.0
        )

        if not pairs_df.empty:
            metrics[
                "json_repair_rate"
            ] = float(
                pairs_df[
                    "json_repaired"
                ].mean()
            )

            metrics[
                "exact_pair_match_rate"
            ] = float(
                pairs_df[
                    "exact_match"
                ].mean()
            )

            metrics[
                "mean_finding_accuracy"
            ] = float(
                pairs_df[
                    "finding_accuracy"
                ].mean()
            )

            metrics[
                "mean_inference_seconds"
            ] = float(
                pairs_df[
                    "elapsed_seconds"
                ].mean()
            )

            metrics[
                "max_gpu_peak_gb"
            ] = float(
                pairs_df[
                    "gpu_peak_gb"
                ]
                .dropna()
                .max()
            )

        metrics_path = (
            output_dir
            / "metrics.json"
        )

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=2,
            ),
            encoding="utf-8",
        )

        disagreements = findings_df[
            ~findings_df[
                "correct"
            ]
        ].copy()

        disagreements.to_csv(
            output_dir
            / "disagreements.csv",
            index=False,
        )

        return metrics