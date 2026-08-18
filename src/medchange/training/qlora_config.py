from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalQLoRAConfig:
    model_name: str = (
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    train_jsonl: str = (
        "data/nih/qlora_smoke/train.jsonl"
    )

    validation_jsonl: str = (
        "data/nih/qlora_smoke/validation.jsonl"
    )

    output_dir: str = (
        "models/qlora/medchange_temporal_smoke"
    )

    seed: int = 42

    num_train_epochs: float = 1.0

    max_steps: int | None = 20

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    gradient_accumulation_steps: int = 4

    learning_rate: float = 2e-4

    warmup_ratio: float = 0.03

    logging_steps: int = 1
    eval_steps: int = 5
    save_steps: int = 5

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    gradient_checkpointing: bool = True

    max_length: int = 2048

    min_pixels: int = 224 * 224
    max_pixels: int = 224 * 224

    use_bf16: bool = False
    use_fp16: bool = True