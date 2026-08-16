from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiomedCLIPConfig:
    """
    Configuration for the BiomedCLIP baseline.
    """

    model_name: str = (
        "hf-hub:"
        "microsoft/"
        "BiomedCLIP-PubMedBERT_256-"
        "vit_base_patch16_224"
    )

    device: str = "cuda"

    precision: str = "fp16"

    context_length: int = 256

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.device not in {
            "cuda",
            "cpu",
        }:
            raise ValueError(
                f"Unsupported device: {self.device}"
            )

        if self.precision not in {
            "fp16",
            "fp32",
        }:
            raise ValueError(
                f"Unsupported precision: {self.precision}"
            )