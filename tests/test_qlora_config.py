import pytest

from medchange.training.qlora_config import (
    TemporalQLoRAConfig,
)


def test_default_smoke_config():
    config = (
        TemporalQLoRAConfig()
    )

    assert (
        config.model_name
        == "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    assert (
        config.per_device_train_batch_size
        == 1
    )

    assert (
        config.gradient_accumulation_steps
        == 4
    )

    assert (
        config.lora_r
        == 8
    )

    assert (
        config.max_steps
        == 20
    )