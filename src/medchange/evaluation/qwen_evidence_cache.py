from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)
from medchange.evaluation.qwen_temporal_metrics import (
    TARGET_FINDINGS,
    prediction_to_confidence_map,
    prediction_to_state_map,
)
from medchange.inference.temporal_pipeline import (
    TemporalQwenPipeline,
)


class QwenTemporalEvidenceCache:
    """
    Run Qwen once over a temporal cohort and persist results
    after every pair.

    Existing successful pair IDs are skipped automatically,
    allowing interrupted runs to resume.
    """

    def __init__(
        self,
        dataset_root: str | Path,
    ) -> None:

        self.resolver = NIHImageResolver(
            dataset_root
        )

        self.resolver.build_index()

        print(
            f"Indexed {self.resolver.num_images:,} "
            "NIH images."
        )

        self.pipeline = (
            TemporalQwenPipeline()
        )

    @staticmethod
    def _load_existing(
        path: Path,
    ) -> pd.DataFrame:

        if not path.exists():
            return pd.DataFrame()

        try:
            dataframe = pd.read_csv(
                path
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        return dataframe

    @staticmethod
    def _append_checkpoint(
        row: dict[str, Any],
        path: Path,
    ) -> None:

        dataframe = pd.DataFrame(
            [row]
        )

        write_header = (
            not path.exists()
            or path.stat().st_size == 0
        )

        dataframe.to_csv(
            path,
            mode="a",
            header=write_header,
            index=False,
        )

    def run(
        self,
        dataframe: pd.DataFrame,
        output_dir: str | Path,
        seed: int = 42,
    ) -> dict[str, Any]:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pair_cache_path = (
            output_dir
            / "qwen_pair_cache.csv"
        )

        finding_cache_path = (
            output_dir
            / "qwen_finding_evidence.csv"
        )

        failure_path = (
            output_dir
            / "failures.csv"
        )

        metadata_path = (
            output_dir
            / "cache_metadata.json"
        )

        existing_pairs = (
            self._load_existing(
                pair_cache_path
            )
        )

        if (
            not existing_pairs.empty
            and "pair_id"
            in existing_pairs.columns
        ):
            completed = set(
                existing_pairs[
                    "pair_id"
                ]
                .astype(str)
                .tolist()
            )

        else:
            completed = set()

        print()
        print("=" * 90)
        print("M4.5.1 — QWEN TEMPORAL EVIDENCE CACHE")
        print("=" * 90)
        print(
            f"Requested pairs : {len(dataframe)}"
        )
        print(
            f"Already cached  : {len(completed)}"
        )
        print(
            f"Remaining       : "
            f"{len(dataframe) - len(completed)}"
        )
        print("=" * 90)

        run_start = time.perf_counter()

        processed_this_run = 0
        failed_this_run = 0

        for position, (_, row) in enumerate(
            dataframe.iterrows(),
            start=1,
        ):
            pair_id = str(
                row["pair_id"]
            )

            if pair_id in completed:
                print(
                    f"[{position}/{len(dataframe)}] "
                    f"SKIP {pair_id}"
                )

                continue

            print()
            print(
                f"[{position}/{len(dataframe)}] "
                f"RUN {pair_id}"
            )

            prior_index = str(
                row["prior_image_index"]
            )

            current_index = str(
                row["current_image_index"]
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
                        prior_image_path=prior_path,
                        current_image_path=current_path,
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

                state_map = (
                    prediction_to_state_map(
                        result.prediction
                    )
                )

                confidence_map = (
                    prediction_to_confidence_map(
                        result.prediction
                    )
                )

                pair_row = {
                    "pair_id": pair_id,
                    "patient_id": row[
                        "patient_id"
                    ],
                    "fusion_category": row.get(
                        "fusion_category",
                        "unknown",
                    ),
                    "prior_image_index": (
                        prior_index
                    ),
                    "current_image_index": (
                        current_index
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
                    "status": "success",
                }

                self._append_checkpoint(
                    pair_row,
                    pair_cache_path,
                )

                for finding in (
                    TARGET_FINDINGS
                ):
                    ground_truth_column = (
                        f"{finding}_temporal"
                    )

                    if (
                        ground_truth_column
                        not in row.index
                    ):
                        raise ValueError(
                            "Missing ground-truth "
                            f"column: "
                            f"{ground_truth_column}"
                        )

                    ground_truth = str(
                        row[
                            ground_truth_column
                        ]
                    )

                    qwen_state = (
                        state_map.get(
                            finding,
                            "uncertain",
                        )
                    )

                    qwen_confidence = float(
                        confidence_map.get(
                            finding,
                            0.0,
                        )
                    )

                    finding_row = {
                        "pair_id": pair_id,
                        "patient_id": row[
                            "patient_id"
                        ],
                        "fusion_category": (
                            row.get(
                                "fusion_category",
                                "unknown",
                            )
                        ),
                        "finding": finding,
                        "ground_truth": (
                            ground_truth
                        ),
                        "qwen_state": (
                            qwen_state
                        ),
                        "qwen_confidence": (
                            qwen_confidence
                        ),
                        "qwen_correct": (
                            qwen_state
                            == ground_truth
                        ),
                        "json_repaired": (
                            result.json_repaired
                        ),
                    }

                    self._append_checkpoint(
                        finding_row,
                        finding_cache_path,
                    )

                completed.add(
                    pair_id
                )

                processed_this_run += 1

                print(
                    "  success | "
                    f"{result.metrics.elapsed_seconds:.2f}s | "
                    "repair="
                    f"{result.json_repaired}"
                )

            except Exception as exc:
                failed_this_run += 1

                failure_row = {
                    "pair_id": pair_id,
                    "patient_id": row.get(
                        "patient_id",
                        "",
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }

                self._append_checkpoint(
                    failure_row,
                    failure_path,
                )

                print(
                    f"  FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        elapsed = (
            time.perf_counter()
            - run_start
        )

        final_pairs = (
            self._load_existing(
                pair_cache_path
            )
        )

        final_findings = (
            self._load_existing(
                finding_cache_path
            )
        )

        successful_pairs = (
            int(
                final_pairs[
                    "pair_id"
                ].nunique()
            )
            if not final_pairs.empty
            else 0
        )

        metadata = {
            "milestone": "M4.5.1",
            "model": (
                "Qwen/Qwen2.5-VL-3B-Instruct"
            ),
            "quantization": "4-bit",
            "seed": seed,
            "requested_pairs": int(
                len(dataframe)
            ),
            "successful_cached_pairs": (
                successful_pairs
            ),
            "processed_this_run": (
                processed_this_run
            ),
            "failed_this_run": (
                failed_this_run
            ),
            "cached_finding_rows": int(
                len(final_findings)
            ),
            "expected_finding_rows": int(
                successful_pairs
                * len(TARGET_FINDINGS)
            ),
            "target_findings": (
                TARGET_FINDINGS
            ),
            "run_elapsed_seconds": (
                elapsed
            ),
        }

        if not final_pairs.empty:
            metadata[
                "mean_inference_seconds"
            ] = float(
                final_pairs[
                    "elapsed_seconds"
                ].mean()
            )

            metadata[
                "max_gpu_peak_gb"
            ] = float(
                final_pairs[
                    "gpu_peak_gb"
                ].max()
            )

            metadata[
                "json_repair_rate"
            ] = float(
                final_pairs[
                    "json_repaired"
                ].mean()
            )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        return metadata