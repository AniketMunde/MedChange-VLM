from __future__ import annotations

from pathlib import Path

from medchange.inference.structured_output import (
    parse_vlm_response,
)
from medchange.models.vlm.qwen_vl import (
    QwenVLM,
)
from medchange.schemas import (
    MedChangePrediction,
)


class MedChangePipeline:
    """
    High-level MedChange-VLM inference pipeline.
    """

    def __init__(
        self,
        vlm: QwenVLM | None = None,
    ) -> None:

        self.vlm = (
            vlm
            if vlm is not None
            else QwenVLM()
        )

    def analyze(
        self,
        image_path: str | Path,
        study_id: str,
        question: str | None = None,
    ) -> MedChangePrediction:

        raw_output = (
            self.vlm.analyze_image(
                image_path=image_path,
                question=question,
            )
        )

        return parse_vlm_response(
            raw_output=raw_output,
            study_id=study_id,
        )