from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TemporalState = Literal[
    "absent",
    "new",
    "persistent",
    "resolved",
    "uncertain",
]


AgreementLevel = Literal[
    "high",
    "partial",
    "conflict",
    "unavailable",
]


UncertaintyLevel = Literal[
    "low",
    "moderate",
    "high",
]


OverallChange = Literal[
    "improved",
    "worsened",
    "mixed",
    "stable",
    "uncertain",
]


class FindingEvidence(BaseModel):
    finding: str

    final_state: TemporalState

    biomedclip_state: TemporalState | None = None
    qwen_state: TemporalState | None = None


    biomedclip_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    qwen_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    agreement: AgreementLevel

    uncertainty: UncertaintyLevel

    evidence: str | None = None

    decision_reason: str | None = None

    requires_review: bool = False


class UnifiedTemporalResult(BaseModel):
    pair_id: str

    prior_study_id: str
    current_study_id: str

    findings: list[
        FindingEvidence
    ]

    overall_change: OverallChange

    uncertainty: UncertaintyLevel

    requires_review: bool

    summary: str