from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Study(BaseModel):
    """
    Represents a single radiographic study.

    The schema intentionally stores metadata separately from model
    predictions so that dataset ingestion and inference remain decoupled.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        min_length=1,
        description="De-identified patient identifier.",
    )

    study_id: str = Field(
        ...,
        min_length=1,
        description="Unique radiographic study identifier.",
    )

    image_path: Path = Field(
        ...,
        description="Path to the study image.",
    )

    study_datetime: Optional[datetime] = Field(
        default=None,
        description="Acquisition date/time when available.",
    )

    report: Optional[str] = Field(
        default=None,
        description="Associated radiology report when available.",
    )


class LongitudinalStudyPair(BaseModel):
    """
    Previous/current pair belonging to the same patient.

    This will later become the primary input representation for
    temporal change reasoning.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    patient_id: str = Field(
        ...,
        min_length=1,
    )

    prior: Study
    current: Study

    time_delta_days: Optional[float] = Field(
        default=None,
        ge=0,
    )

    def validate_same_patient(self) -> None:
        """
        Ensure both studies belong to the same patient.
        """

        if self.prior.patient_id != self.current.patient_id:
            raise ValueError(
                "Prior and current studies must belong to the same patient."
            )

        if self.patient_id != self.prior.patient_id:
            raise ValueError(
                "Longitudinal pair patient_id does not match study patient_id."
            )