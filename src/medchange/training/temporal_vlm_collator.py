from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


class TemporalVLMCollator:
    """
    Multimodal collator for MedChange temporal QLoRA.

    Important:
    Loss is computed ONLY on assistant response tokens.

    System prompt, user prompt, image placeholder tokens,
    and padding are masked with -100.
    """

    def __init__(
        self,
        processor,
        *,
        max_length: int = 2048,
    ) -> None:
        self.processor = processor
        self.max_length = max_length

        # Keep prompt/full sequence alignment predictable.
        if hasattr(
            self.processor,
            "tokenizer",
        ):
            self.processor.tokenizer.padding_side = "right"

    def _load_images(
        self,
        image_paths: list[str],
    ) -> list[Image.Image]:

        if len(image_paths) != 2:
            raise ValueError(
                "Temporal VLM training requires "
                "exactly two images per example."
            )

        images = []

        for path in image_paths:
            image_path = Path(
                path
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Training image not found: "
                    f"{image_path}"
                )

            image = (
                Image.open(
                    image_path
                )
                .convert(
                    "RGB"
                )
            )

            images.append(
                image
            )

        return images

    @staticmethod
    def _build_assistant_only_labels(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        pad_token_id: int | None,
    ) -> torch.Tensor:
        """
        Create labels where only assistant tokens
        contribute to causal-LM loss.
        """

        labels = (
            input_ids
            .clone()
        )

        batch_size = (
            input_ids.shape[0]
        )

        if len(
            prompt_lengths
        ) != batch_size:
            raise ValueError(
                "prompt_lengths must match "
                "batch size."
            )

        for index, prompt_length in enumerate(
            prompt_lengths
        ):
            sequence_length = int(
                attention_mask[
                    index
                ].sum().item()
            )

            prompt_length = min(
                int(
                    prompt_length
                ),
                sequence_length,
            )

            # Mask everything before assistant response.
            labels[
                index,
                :prompt_length,
            ] = -100

            # Mask padding / tokens after sequence.
            labels[
                index,
                sequence_length:
            ] = -100

        if pad_token_id is not None:
            labels[
                input_ids
                == pad_token_id
            ] = -100

        return labels

    def __call__(
        self,
        examples: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        full_texts = []
        prompt_texts = []

        image_batches = []

        for example in examples:

            messages = (
                example[
                    "messages"
                ]
            )

            if len(
                messages
            ) < 3:
                raise ValueError(
                    "Expected system, user, "
                    "and assistant messages."
                )

            # ------------------------------
            # Full training conversation
            # ------------------------------

            full_text = (
                self.processor
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            # ------------------------------
            # Prompt without assistant target
            # ------------------------------

            prompt_messages = (
                messages[
                    :-1
                ]
            )

            prompt_text = (
                self.processor
                .apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            full_texts.append(
                full_text
            )

            prompt_texts.append(
                prompt_text
            )

            image_batches.append(
                self._load_images(
                    example[
                        "images"
                    ]
                )
            )

        # ====================================================
        # FULL INPUT
        # ====================================================

        batch = (
            self.processor(
                text=full_texts,
                images=image_batches,
                padding=True,
                truncation=True,
                max_length=(
                    self.max_length
                ),
                return_tensors="pt",
            )
        )

        # ====================================================
        # PROMPT INPUT
        # Used only to determine where assistant starts.
        # ====================================================

        prompt_batch = (
            self.processor(
                text=prompt_texts,
                images=image_batches,
                padding=True,
                truncation=True,
                max_length=(
                    self.max_length
                ),
                return_tensors="pt",
            )
        )

        prompt_lengths = [
            int(
                mask.sum().item()
            )
            for mask
            in prompt_batch[
                "attention_mask"
            ]
        ]

        pad_token_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        labels = (
            self._build_assistant_only_labels(
                input_ids=(
                    batch[
                        "input_ids"
                    ]
                ),

                attention_mask=(
                    batch[
                        "attention_mask"
                    ]
                ),

                prompt_lengths=(
                    prompt_lengths
                ),

                pad_token_id=(
                    pad_token_id
                ),
            )
        )

        batch[
            "labels"
        ] = labels

        # Fail early if truncation accidentally removed
        # every assistant target token.
        trainable_tokens = int(
            (
                labels
                != -100
            )
            .sum()
            .item()
        )

        if trainable_tokens == 0:
            raise ValueError(
                "No assistant tokens remain after "
                "masking/truncation. Increase max_length."
            )

        return batch