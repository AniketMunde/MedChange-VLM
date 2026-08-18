from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from medchange.api.dependencies import (
    TARGET_FINDINGS,
)
from medchange.inference.temporal_pipeline import (
    TemporalQwenPipeline,
)
from medchange.models.temporal.ablation_features import (
    build_ablation_vector,
)
from medchange.models.temporal.features import (
    build_temporal_embedding_features,
)
from medchange.models.vision import (
    BiomedCLIP,
)
from medchange.models.vlm.config import (
    VLMConfig,
)
from medchange.models.vlm.qwen_vl import (
    QwenVLM,
)
from medchange.reasoning.unified_pipeline import (
    build_unified_temporal_result,
)
from medchange.reporting.report_builder import (
    build_longitudinal_report,
)
from medchange.safety.config import (
    SafetyPolicyConfig,
)


def _load_classifier_artifact(
    classifier_dir: Path,
    finding: str,
) -> dict:
    path = (
        classifier_dir
        / f"{finding}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing temporal classifier: "
            f"{path}"
        )

    with path.open(
        "rb"
    ) as file:
        artifact = pickle.load(
            file
        )

    required = {
        "finding",
        "feature_set",
        "classifier",
    }

    missing = (
        required
        - set(
            artifact.keys()
        )
    )

    if missing:
        raise ValueError(
            f"Classifier artifact for "
            f"{finding} is missing: "
            f"{sorted(missing)}"
        )

    return artifact


def _normalize_finding(
    finding: str,
) -> str:
    return (
        str(finding)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    )


class MedChangeService:
    def __init__(
        self,
        classifier_dir: Path,
    ) -> None:
        self.classifier_dir = (
            classifier_dir
        )

        self._biomedclip = None
        self._qwen = None
        self._qwen_pipeline = None

        self._classifier_artifacts = {
            finding:
                _load_classifier_artifact(
                    classifier_dir,
                    finding,
                )
            for finding
            in TARGET_FINDINGS
        }

    def _get_biomedclip(
        self,
    ) -> BiomedCLIP:
        if self._biomedclip is None:
            print(
                "API: loading BiomedCLIP..."
            )

            self._biomedclip = (
                BiomedCLIP()
            )

        return self._biomedclip

    def _get_qwen_pipeline(
        self,
    ) -> TemporalQwenPipeline:
        if (
            self._qwen_pipeline
            is None
        ):
            print(
                "API: loading Qwen2.5-VL..."
            )

            config = VLMConfig()

            self._qwen = QwenVLM(
                config=config
            )

            self._qwen_pipeline = (
                TemporalQwenPipeline(
                    vlm=self._qwen
                )
            )

        return self._qwen_pipeline

    def _run_biomedclip(
        self,
        prior_path: Path,
        current_path: Path,
    ) -> dict:
        model = (
            self._get_biomedclip()
        )

        prior_embedding = (
            model.encode_image(
                prior_path
            )
        )

        current_embedding = (
            model.encode_image(
                current_path
            )
        )

        temporal_features = (
            build_temporal_embedding_features(
                prior_embedding,
                current_embedding,
            )
        )

        predictions = {}

        for finding in (
            TARGET_FINDINGS
        ):
            artifact = (
                self._classifier_artifacts[
                    finding
                ]
            )

            classifier = (
                artifact[
                    "classifier"
                ]
            )

            feature_set = (
                artifact[
                    "feature_set"
                ]
            )

            feature_vector = (
                build_ablation_vector(
                    temporal_features,
                    feature_set,
                )
                .reshape(
                    1,
                    -1,
                )
            )

            state = str(
                classifier.predict(
                    feature_vector
                )[0]
            )

            confidence = None

            if (
                hasattr(
                    classifier,
                    "model",
                )
                and hasattr(
                    classifier.model,
                    "predict_proba",
                )
            ):
                scaled = (
                    classifier.scaler
                    .transform(
                        feature_vector
                    )
                )

                probabilities = (
                    classifier.model
                    .predict_proba(
                        scaled
                    )[0]
                )

                confidence = float(
                    np.max(
                        probabilities
                    )
                )

            predictions[
                finding
            ] = {
                "state": (
                    state
                ),

                "confidence": (
                    confidence
                ),
            }

        return predictions

    def _run_qwen(
        self,
        *,
        prior_path: Path,
        current_path: Path,
        pair_id: str,
        prior_study_id: str,
        current_study_id: str,
    ) -> tuple[
        dict,
        dict,
        dict,
    ]:
        pipeline = (
            self._get_qwen_pipeline()
        )

        result = (
            pipeline
            .analyze_pair_detailed(
                prior_image_path=(
                    prior_path
                ),

                current_image_path=(
                    current_path
                ),

                pair_id=(
                    pair_id
                ),

                prior_study_id=(
                    prior_study_id
                ),

                current_study_id=(
                    current_study_id
                ),

                save_raw_output=False,
            )
        )

        predictions = {}
        explanations = {}

        for finding in (
            result.prediction.findings
        ):
            name = (
                _normalize_finding(
                    finding.finding
                )
            )

            if name not in (
                TARGET_FINDINGS
            ):
                continue

            predictions[
                name
            ] = {
                "state": (
                    finding.change
                ),

                "confidence": float(
                    finding.confidence
                ),
            }

            explanations[
                name
            ] = (
                finding.evidence
            )

        metrics = {
            "elapsed_seconds": (
                result.metrics
                .elapsed_seconds
            ),

            "gpu_peak_gb": (
                result.metrics
                .gpu_peak_allocated_gb
            ),

            "json_repaired": (
                result.json_repaired
            ),
        }

        return (
            predictions,
            explanations,
            metrics,
        )

    def analyze_pair(
        self,
        *,
        prior_path: Path,
        current_path: Path,
        pair_id: str,
        prior_study_id: str,
        current_study_id: str,
        safety_config: SafetyPolicyConfig,
    ) -> dict:
        start = (
            time.perf_counter()
        )

        biomedclip_predictions = (
            self._run_biomedclip(
                prior_path=(
                    prior_path
                ),
                current_path=(
                    current_path
                ),
            )
        )

        (
            qwen_predictions,
            qwen_explanations,
            qwen_metrics,
        ) = (
            self._run_qwen(
                prior_path=(
                    prior_path
                ),

                current_path=(
                    current_path
                ),

                pair_id=(
                    pair_id
                ),

                prior_study_id=(
                    prior_study_id
                ),

                current_study_id=(
                    current_study_id
                ),
            )
        )

        unified = (
            build_unified_temporal_result(
                pair_id=(
                    pair_id
                ),

                prior_study_id=(
                    prior_study_id
                ),

                current_study_id=(
                    current_study_id
                ),

                biomedclip_predictions=(
                    biomedclip_predictions
                ),

                qwen_predictions=(
                    qwen_predictions
                ),

                qwen_explanations=(
                    qwen_explanations
                ),

                safety_config=(
                    safety_config
                ),
            )
        )

        report = (
            build_longitudinal_report(
                unified
            )
        )

        total_elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "result": (
                unified
            ),

            "report": (
                report
            ),

            "biomedclip": (
                biomedclip_predictions
            ),

            "qwen": (
                qwen_predictions
            ),

            "qwen_metrics": (
                qwen_metrics
            ),

            "total_elapsed_seconds": (
                total_elapsed
            ),
        }