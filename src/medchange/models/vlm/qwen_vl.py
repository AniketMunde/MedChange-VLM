from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from medchange.models.vlm.config import (
    VLMConfig,
)
from medchange.models.vlm.prompts import (
    SYSTEM_PROMPT,
    build_single_image_prompt,
)


class QwenVLM:
    """
    VRAM-aware wrapper around Qwen2.5-VL.

    The initial baseline supports single-image chest-X-ray analysis.
    """

    def __init__(
        self,
        config: Optional[VLMConfig] = None,
    ) -> None:

        self.config = config or VLMConfig()

        self.model = None
        self.processor = None

    @staticmethod
    def _resolve_dtype(
        dtype_name: str,
    ) -> torch.dtype:

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        if dtype_name not in mapping:
            raise ValueError(
                f"Unsupported dtype: {dtype_name}"
            )

        return mapping[dtype_name]

    def _build_quantization_config(
        self,
    ) -> BitsAndBytesConfig | None:

        if not self.config.load_in_4bit:
            return None

        compute_dtype = self._resolve_dtype(
            self.config.compute_dtype
        )

        return BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=(
                self.config.quant_type
            ),

            bnb_4bit_use_double_quant=(
                self.config.use_double_quant
            ),

            bnb_4bit_compute_dtype=(
                compute_dtype
            ),
        )

    def load(self) -> None:
        """
        Load processor and quantized Qwen model.
        """

        if self.model is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the current "
                "MedChange-VLM baseline."
            )

        min_pixels = (
            self.config.min_visual_tokens
            * 28
            * 28
        )

        max_pixels = (
            self.config.max_visual_tokens
            * 28
            * 28
        )

        print(
            f"Loading processor: "
            f"{self.config.model_name}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.config.model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                trust_remote_code=(
                    self.config.trust_remote_code
                ),
            )
        )

        quantization_config = (
            self._build_quantization_config()
        )

        print(
            "Loading Qwen2.5-VL in 4-bit mode..."
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                self.config.model_name,

                quantization_config=(
                    quantization_config
                ),

                device_map=(
                    self.config.device_map
                ),

                torch_dtype=(
                    self._resolve_dtype(
                        self.config.compute_dtype
                    )
                ),

                trust_remote_code=(
                    self.config.trust_remote_code
                ),
            )
        )

        self.model.eval()

        print(
            "Qwen2.5-VL loaded successfully."
        )

        if torch.cuda.is_available():
            allocated = (
                torch.cuda.memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                / 1024**3
            )

            print(
                f"CUDA allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"CUDA reserved : "
                f"{reserved:.2f} GB"
            )

    def analyze_image(
        self,
        image_path: str | Path,
        question: str | None = None,
    ) -> str:
        """
        Run inference on one chest radiograph.

        Returns raw generated model text.
        """

        if self.model is None:
            self.load()

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        prompt = build_single_image_prompt(
            question
        )

        image = Image.open(
            path
        ).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(
                messages
            )
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        target_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: (
                value.to(target_device)
                if hasattr(value, "to")
                else value
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        self.config.max_new_tokens
                    ),
                    do_sample=False,
                    use_cache=True,
                )
            )

        generated_ids_trimmed = [
            output_ids[
                len(input_ids):
            ]
            for input_ids, output_ids
            in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        output = (
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

        return output.strip()

    def unload(self) -> None:
        """
        Release GPU memory.
        """

        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()