from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


TemporalChange = Literal[
    "new",
    "resolved",
    "persistent",
    "absent",
    "uncertain",
]


OverallTemporalChange = Literal[
    "improved",
    "worsened",
    "mixed",
    "stable",
    "uncertain",
]


class TemporalFinding(BaseModel):
    finding: str

    prior_status: Literal[
        "present",
        "absent",
        "uncertain",
    ]

    current_status: Literal[
        "present",
        "absent",
        "uncertain",
    ]

    change: TemporalChange

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: str


class TemporalPrediction(BaseModel):
    pair_id: str

    prior_study_id: str
    current_study_id: str

    findings: list[
        TemporalFinding
    ]

    overall_change: OverallTemporalChange

    summary: str