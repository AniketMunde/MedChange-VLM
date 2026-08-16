from __future__ import annotations

from pathlib import Path

import open_clip
import torch
from PIL import Image

from medchange.models.vision.config import (
    BiomedCLIPConfig,
)
from medchange.models.vision.prompts import (
    build_prompt_pair,
)


class BiomedCLIP:
    """
    Biomedical image-text encoder based on Microsoft's BiomedCLIP.

    Supports:
    - image embeddings
    - text embeddings
    - cached pathology text prototypes
    - zero-shot abnormality scoring
    """

    def __init__(
        self,
        config: BiomedCLIPConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else BiomedCLIPConfig()
        )

        self.model = None
        self.preprocess = None
        self.tokenizer = None

        self._finding_text_cache: dict[
            str,
            torch.Tensor,
        ] = {}

    def load(self) -> None:
        """
        Load BiomedCLIP and its preprocessing pipeline.
        """

        if self.model is not None:
            return

        if (
            self.config.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        print(
            f"Loading BiomedCLIP: "
            f"{self.config.model_name}"
        )

        (
            self.model,
            _,
            self.preprocess,
        ) = open_clip.create_model_and_transforms(
            self.config.model_name
        )

        self.tokenizer = open_clip.get_tokenizer(
            self.config.model_name
        )

        self.model = self.model.to(
            self.config.device
        )

        if (
            self.config.precision == "fp16"
            and self.config.device == "cuda"
        ):
            self.model = self.model.half()

        self.model.eval()

        print(
            "BiomedCLIP loaded successfully."
        )

    def encode_image(
            self,
            image_input: str | Path | Image.Image,
    ) -> torch.Tensor:
        """
        Produce a normalized biomedical image embedding.

        Supports either:
        - filesystem image paths
        - already decoded PIL images

        This allows both local datasets and streamed
        Hugging Face datasets to use the same model.
        """

        if self.model is None:
            self.load()

        if isinstance(
                image_input,
                Image.Image,
        ):
            image = image_input.convert(
                "RGB"
            )

        else:
            path = Path(
                image_input
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            with Image.open(
                    path
            ) as loaded_image:
                image = loaded_image.convert(
                    "RGB"
                )

        image_tensor = (
            self.preprocess(
                image
            )
            .unsqueeze(0)
            .to(
                self.config.device
            )
        )

        if (
                self.config.precision == "fp16"
                and self.config.device == "cuda"
        ):
            image_tensor = (
                image_tensor.half()
            )

        with torch.inference_mode():
            image_features = (
                self.model.encode_image(
                    image_tensor
                )
            )

        image_features = (
                image_features
                / image_features.norm(
            dim=-1,
            keepdim=True,
        )
        )

        return image_features

    def encode_text(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Produce normalized biomedical text embeddings.
        """

        if self.model is None:
            self.load()

        tokens = self.tokenizer(
            texts,
            context_length=(
                self.config.context_length
            ),
        ).to(
            self.config.device
        )

        with torch.inference_mode():
            text_features = (
                self.model.encode_text(
                    tokens
                )
            )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        return text_features

    def get_finding_text_features(
        self,
        finding: str,
    ) -> torch.Tensor:
        """
        Return cached positive/negative text embeddings
        for a pathology finding.
        """

        normalized_finding = (
            finding.strip().lower()
        )

        if (
            normalized_finding
            in self._finding_text_cache
        ):
            return self._finding_text_cache[
                normalized_finding
            ]

        positive_prompt, negative_prompt = (
            build_prompt_pair(
                normalized_finding
            )
        )

        text_features = self.encode_text(
            [
                positive_prompt,
                negative_prompt,
            ]
        )

        self._finding_text_cache[
            normalized_finding
        ] = text_features

        return text_features

    def warm_text_cache(
        self,
        findings: list[str],
    ) -> None:
        """
        Precompute all pathology text embeddings.
        """

        for finding in findings:
            self.get_finding_text_features(
                finding
            )

    def clear_text_cache(self) -> None:
        self._finding_text_cache.clear()

    def score_finding_from_embedding(
        self,
        image_features: torch.Tensor,
        finding: str,
    ) -> float:
        """
        Score a finding using an already encoded image.
        """

        text_features = (
            self.get_finding_text_features(
                finding
            )
        )

        logits = (
            100.0
            * image_features
            @ text_features.T
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return float(
            probabilities[0, 0].item()
        )

    def score_finding(
            self,
            image_input: str | Path | Image.Image,
            finding: str,
    ) -> dict[str, float | str]:

        image_features = (
            self.encode_image(
                image_input
            )
        )

        probability = (
            self.score_finding_from_embedding(
                image_features,
                finding,
            )
        )

        return {
            "finding": finding,
            "positive_probability": probability,
            "negative_probability": (
                    1.0 - probability
            ),
        }

    def score_findings(
            self,
            image_input: str | Path | Image.Image,
            findings: list[str],
    ) -> dict[str, float]:
        """
        Score multiple findings efficiently.

        The image is encoded exactly once.
        """

        if not findings:
            return {}

        self.warm_text_cache(
            findings
        )

        image_features = (
            self.encode_image(
                image_input
            )
        )

        scores: dict[str, float] = {}

        for finding in findings:
            scores[finding] = (
                self.score_finding_from_embedding(
                    image_features,
                    finding,
                )
            )

        return scores