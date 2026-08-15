from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMConfig:
    """
    Runtime configuration for the MedChange vision-language model.
    """

    model_name: str = (
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    load_in_4bit: bool = True

    quant_type: str = "nf4"

    use_double_quant: bool = True

    compute_dtype: str = "float16"

    min_visual_tokens: int = 256

    max_visual_tokens: int = 512

    max_new_tokens: int = 512

    device_map: str = "auto"

    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.min_visual_tokens <= 0:
            raise ValueError(
                "min_visual_tokens must be positive."
            )

        if (
            self.max_visual_tokens
            < self.min_visual_tokens
        ):
            raise ValueError(
                "max_visual_tokens must be greater than "
                "or equal to min_visual_tokens."
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )