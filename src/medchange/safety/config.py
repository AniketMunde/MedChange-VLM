from __future__ import annotations

from dataclasses import dataclass


VALID_SAFETY_POLICIES = {
    "strict",
    "change_sensitive",
}


DEFAULT_POLICY = "change_sensitive"


DEFAULT_THRESHOLDS = {
    "strict": 0.60,
    "change_sensitive": 0.80,
}


@dataclass(frozen=True)
class SafetyPolicyConfig:
    policy: str = DEFAULT_POLICY
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.policy not in VALID_SAFETY_POLICIES:
            raise ValueError(
                f"Unsupported safety policy: "
                f"{self.policy}"
            )

        resolved_threshold = (
            DEFAULT_THRESHOLDS[
                self.policy
            ]
            if self.threshold is None
            else float(
                self.threshold
            )
        )

        if not (
            0.0
            <= resolved_threshold
            <= 1.0
        ):
            raise ValueError(
                "Safety threshold must be "
                "between 0 and 1."
            )

        object.__setattr__(
            self,
            "threshold",
            resolved_threshold,
        )