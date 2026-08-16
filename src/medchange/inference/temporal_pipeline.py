from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path

from medchange.inference.logging_utils import (
    save_raw_vlm_output,
)
from medchange.inference.metrics import (
    InferenceMetrics,
    InferenceTimer,
)
from medchange.inference.temporal_structured_output import (
    parse_temporal_vlm_response,
    parse_temporal_vlm_response_with_metadata,
)
from medchange.models.vlm.qwen_vl import (
    QwenVLM,
)
from medchange.models.vlm.temporal_prompts import (
    build_temporal_comparison_prompt,
)
from medchange.schemas_temporal import (
    TemporalPrediction,
)

@dataclass(frozen=True)
class TemporalInferenceResult:
    prediction: TemporalPrediction

    metrics: InferenceMetrics

    raw_output: str

    json_repaired: bool

class TemporalQwenPipeline:
    def __init__(
        self,
        vlm: QwenVLM | None = None,
    ) -> None:

        self.vlm = (
            vlm
            if vlm is not None
            else QwenVLM()
        )

    def analyze_pair(
        self,
        prior_image_path: str | Path,
        current_image_path: str | Path,
        pair_id: str,
        prior_study_id: str,
        current_study_id: str,
        save_raw_output: bool = True,
    ) -> tuple[
        TemporalPrediction,
        InferenceMetrics,
    ]:

        prompt = (
            build_temporal_comparison_prompt()
        )

        timer = InferenceTimer()

        timer.start()

        raw_output = (
            self.vlm.analyze_image_pair(
                prior_image_path=(
                    prior_image_path
                ),

                current_image_path=(
                    current_image_path
                ),

                question=prompt,
            )
        )

        metrics = timer.stop()

        if save_raw_output:
            save_raw_vlm_output(
                output=raw_output,

                study_id=(
                    f"temporal_{pair_id}"
                ),
            )

        prediction = (
            parse_temporal_vlm_response(
                raw_output=raw_output,

                pair_id=pair_id,

                prior_study_id=(
                    prior_study_id
                ),

                current_study_id=(
                    current_study_id
                ),
            )
        )

        return (
            prediction,
            metrics,
        )

    def analyze_pair_detailed(
            self,
            prior_image_path: str | Path,
            current_image_path: str | Path,
            pair_id: str,
            prior_study_id: str,
            current_study_id: str,
            save_raw_output: bool = True,
    ) -> TemporalInferenceResult:

        prompt = (
            build_temporal_comparison_prompt()
        )

        timer = InferenceTimer()

        timer.start()

        raw_output = (
            self.vlm.analyze_image_pair(
                prior_image_path=(
                    prior_image_path
                ),
                current_image_path=(
                    current_image_path
                ),
                question=prompt,
            )
        )

        metrics = timer.stop()

        if save_raw_output:
            save_raw_vlm_output(
                output=raw_output,
                study_id=(
                    f"temporal_{pair_id}"
                ),
            )

        parsed = (
            parse_temporal_vlm_response_with_metadata(
                raw_output=raw_output,
                pair_id=pair_id,
                prior_study_id=prior_study_id,
                current_study_id=current_study_id,
            )
        )

        return TemporalInferenceResult(
            prediction=(
                parsed.prediction
            ),
            metrics=metrics,
            raw_output=raw_output,
            json_repaired=(
                parsed.json_repaired
            ),
        )