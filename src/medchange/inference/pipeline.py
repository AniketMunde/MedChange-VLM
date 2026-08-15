from __future__ import annotations

from pathlib import Path

from medchange.inference.logging_utils import (
    save_raw_vlm_output,
)
from medchange.inference.metrics import (
    InferenceMetrics,
    InferenceTimer,
)
from medchange.inference.structured_output import (
    parse_vlm_response,
)
from medchange.models.vlm.qwen_vl import QwenVLM
from medchange.schemas import MedChangePrediction


class MedChangePipeline:
    def __init__(
        self,
        vlm: QwenVLM | None = None,
    ) -> None:
        self.vlm = vlm if vlm is not None else QwenVLM()

    def analyze(
        self,
        image_path: str | Path,
        study_id: str,
        question: str | None = None,
        save_raw_output: bool = True,
    ) -> tuple[MedChangePrediction, InferenceMetrics]:

        timer = InferenceTimer()
        timer.start()

        raw_output = self.vlm.analyze_image(
            image_path=image_path,
            question=question,
        )

        metrics = timer.stop()

        if save_raw_output:
            save_raw_vlm_output(
                output=raw_output,
                study_id=study_id,
            )

        prediction = parse_vlm_response(
            raw_output=raw_output,
            study_id=study_id,
        )

        return prediction, metrics