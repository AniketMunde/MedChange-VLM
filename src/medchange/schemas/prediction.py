from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TemporalStatus(str, Enum):
    NEW = "new"
    WORSENED = "worsened"
    STABLE = "stable"
    IMPROVED = "improved"
    RESOLVED = "resolved"
    UNCERTAIN = "uncertain"


class BoundingBox(BaseModel):
    """
    Normalized bounding box coordinates.

    Coordinates are expected in the range [0, 1].
    """

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

    def validate_geometry(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max.")

        if self.y_min >= self.y_max:
            raise ValueError("y_min must be smaller than y_max.")


class FindingLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anatomy: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None


class MedicalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)

    present: bool

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    temporal_status: TemporalStatus = TemporalStatus.UNCERTAIN

    location: Optional[FindingLocation] = None


class UncertaintyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    requires_review: bool = True


class VisualAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dependency_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    grounding_supported: Optional[bool] = None


class MedChangePrediction(BaseModel):
    """
    Canonical output schema for MedChange-VLM.

    Model inference, agents, API responses and evaluation code should
    eventually communicate using this representation.
    """

    model_config = ConfigDict(extra="forbid")

    study_id: str

    findings: List[MedicalFinding] = Field(
        default_factory=list,
    )

    comparison: Optional[str] = None

    impression: Optional[str] = None

    uncertainty: Optional[UncertaintyResult] = None

    visual_audit: Optional[VisualAuditResult] = None