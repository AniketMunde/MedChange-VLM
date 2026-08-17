from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from medchange.safety.config import (
    SafetyPolicyConfig,
)


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


@lru_cache(maxsize=1)
def get_classifier_dir() -> Path:
    path = Path(
        "models/temporal"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Temporal classifier directory "
            f"not found: {path}"
        )

    return path


def get_default_safety_config() -> SafetyPolicyConfig:
    return SafetyPolicyConfig(
        policy="change_sensitive",
        threshold=0.80,
    )