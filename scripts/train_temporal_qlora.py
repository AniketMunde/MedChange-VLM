from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
import inspect
import numpy as np
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from medchange.training.qlora_config import (
    TemporalQLoRAConfig,
)
from medchange.training.temporal_vlm_collator import (
    TemporalVLMCollator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M7.2 QLoRA smoke fine-tuning "
            "for MedChange temporal reasoning."
        )
    )

    parser.add_argument(
        "--train-jsonl",
        default=(
            "data/nih/qlora_smoke/train.jsonl"
        ),
    )

    parser.add_argument(
        "--validation-jsonl",
        default=(
            "data/nih/qlora_smoke/validation.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "models/qlora/"
            "medchange_temporal_smoke"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_jsonl(
    path: str | Path,
) -> list[dict]:
    records = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(
                    line
                )
            )

    if not records:
        raise ValueError(
            f"No records found: {path}"
        )

    return records


def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def main() -> None:
    args = parse_args()

    config = (
        TemporalQLoRAConfig(
            train_jsonl=(
                args.train_jsonl
            ),
            validation_jsonl=(
                args.validation_jsonl
            ),
            output_dir=(
                args.output_dir
            ),
            max_steps=(
                args.max_steps
            ),
            seed=(
                args.seed
            ),
        )
    )

    set_seed(
        config.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this "
            "QLoRA smoke experiment."
        )

    print()
    print("=" * 90)
    print(
        "M7.2 — MEDCHANGE TEMPORAL QLoRA SMOKE TRAINING"
    )
    print("=" * 90)

    print(
        f"Model      : {config.model_name}"
    )

    print(
        f"Train data : {config.train_jsonl}"
    )

    print(
        f"Val data   : {config.validation_jsonl}"
    )

    print(
        f"Max steps  : {config.max_steps}"
    )

    print(
        f"Output     : {config.output_dir}"
    )

    print()

    train_records = load_jsonl(
        config.train_jsonl
    )

    validation_records = load_jsonl(
        config.validation_jsonl
    )

    train_dataset = (
        Dataset.from_list(
            train_records
        )
    )

    validation_dataset = (
        Dataset.from_list(
            validation_records
        )
    )

    print(
        f"Train examples      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples : "
        f"{len(validation_dataset)}"
    )

    # ------------------------------------------------------
    # Processor
    # ------------------------------------------------------

    processor = (
        AutoProcessor
        .from_pretrained(
            config.model_name,
            min_pixels=(
                config.min_pixels
            ),
            max_pixels=(
                config.max_pixels
            ),
        )
    )

    if (
        processor.tokenizer.pad_token_id
        is None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    # ------------------------------------------------------
    # 4-bit QLoRA quantization
    # ------------------------------------------------------

    compute_dtype = (
        torch.float16
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_use_double_quant=True,

            bnb_4bit_compute_dtype=(
                compute_dtype
            ),
        )
    )

    print(
        "Loading Qwen2.5-VL in 4-bit..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            config.model_name,

            quantization_config=(
                quantization_config
            ),

            device_map="auto",

            torch_dtype=(
                compute_dtype
            ),
        )
    )

    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Official PEFT QLoRA preparation step.
    model = (
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=(
                config.gradient_checkpointing
            ),
        )
    )

    # ------------------------------------------------------
    # LoRA
    # ------------------------------------------------------

    lora_config = (
        LoraConfig(
            r=(
                config.lora_r
            ),

            lora_alpha=(
                config.lora_alpha
            ),

            lora_dropout=(
                config.lora_dropout
            ),

            bias="none",

            task_type=(
                "CAUSAL_LM"
            ),

            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ------------------------------------------------------
    # Collator
    # ------------------------------------------------------

    collator = (
        TemporalVLMCollator(
            processor=processor,
            max_length=(
                config.max_length
            ),
        )
    )

    # ------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------

    import inspect

    training_kwargs = {
        "output_dir": (
            config.output_dir
        ),

        "num_train_epochs": (
            config.num_train_epochs
        ),

        "max_steps": (
            config.max_steps
        ),

        "per_device_train_batch_size": (
            config.per_device_train_batch_size
        ),

        "per_device_eval_batch_size": (
            config.per_device_eval_batch_size
        ),

        "gradient_accumulation_steps": (
            config.gradient_accumulation_steps
        ),

        "learning_rate": (
            config.learning_rate
        ),

        # For the smoke run we do not need warmup.
        "warmup_steps": 0,

        "logging_steps": (
            config.logging_steps
        ),

        "eval_steps": (
            config.eval_steps
        ),

        "save_steps": (
            config.save_steps
        ),

        "save_total_limit": 2,

        "fp16": (
            config.use_fp16
        ),

        "bf16": (
            config.use_bf16
        ),

        "gradient_checkpointing": (
            config.gradient_checkpointing
        ),

        "optim": (
            "paged_adamw_8bit"
        ),

        "report_to": "none",

        "remove_unused_columns": False,

        "dataloader_num_workers": 0,

        "seed": (
            config.seed
        ),

        "logging_first_step": True,

        "load_best_model_at_end": False,
    }

    # ============================================================
    # TRANSFORMERS VERSION COMPATIBILITY
    # ============================================================

    training_signature = (
        inspect.signature(
            TrainingArguments.__init__
        )
    )

    supported_arguments = set(
        training_signature.parameters.keys()
    )

    # Different Transformers releases use either
    # "eval_strategy" or "evaluation_strategy".
    if (
            "eval_strategy"
            in supported_arguments
    ):
        training_kwargs[
            "eval_strategy"
        ] = "steps"

    elif (
            "evaluation_strategy"
            in supported_arguments
    ):
        training_kwargs[
            "evaluation_strategy"
        ] = "steps"

    # Same idea for save strategy.
    if (
            "save_strategy"
            in supported_arguments
    ):
        training_kwargs[
            "save_strategy"
        ] = "steps"

    # Remove anything unsupported by the installed
    # Transformers version.
    training_kwargs = {
        key: value
        for key, value
        in training_kwargs.items()
        if key
           in supported_arguments
    }

    print()
    print(
        "TrainingArguments supported configuration:"
    )

    for key in sorted(
            training_kwargs
    ):
        print(
            f"  {key} = "
            f"{training_kwargs[key]}"
        )

    training_args = TrainingArguments(
        **training_kwargs
    )

    trainer = Trainer(
        model=model,

        args=(
            training_args
        ),

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            validation_dataset
        ),

        data_collator=(
            collator
        ),
    )

    print()
    print(
        "Starting QLoRA smoke training..."
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_result = (
        trainer.train()
    )
    print()
    print(
        "Running final validation..."
    )

    eval_metrics = (
        trainer.evaluate()
    )

    print(
        "Final validation metrics:"
    )

    for key, value in (
            eval_metrics.items()
    ):
        print(
            f"  {key}: {value}"
        )
    print()
    print(
        "Training complete."
    )

    trainer.save_model(
        config.output_dir
    )

    processor.save_pretrained(
        config.output_dir
    )

    metrics = {
        "training": dict(
            train_result.metrics
        ),

        "validation": dict(
            eval_metrics
        ),

        "train_examples": len(
            train_dataset
        ),

        "validation_examples": len(
            validation_dataset
        ),

        "max_steps": (
            config.max_steps
        ),

        "seed": (
            config.seed
        ),

        "lora": {
            "rank": (
                config.lora_r
            ),

            "alpha": (
                config.lora_alpha
            ),

            "dropout": (
                config.lora_dropout
            ),

            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },
    }

    metrics[
        "train_examples"
    ] = len(
        train_dataset
    )

    metrics[
        "validation_examples"
    ] = len(
        validation_dataset
    )

    metrics[
        "max_steps"
    ] = (
        config.max_steps
    )

    metrics_path = (
        Path(
            config.output_dir
        )
        / "smoke_train_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 90
    )

    print(
        "M7.2 SMOKE TRAINING COMPLETE"
    )

    print(
        f"Adapter : {config.output_dir}"
    )

    print(
        f"Metrics : {metrics_path}"
    )

    if torch.cuda.is_available():
        print(
            "CUDA allocated : "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            "CUDA reserved  : "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

        print(
            "CUDA peak      : "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()