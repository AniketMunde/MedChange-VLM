from __future__ import annotations

from pydantic import BaseModel


class ReportFinding(BaseModel):
    finding: str
    state: str
    agreement: str
    uncertainty: str
    evidence: str | None = None
    requires_review: bool = False


class LongitudinalReport(BaseModel):
    pair_id: str
    prior_study_id: str
    current_study_id: str

    new_findings: list[ReportFinding]
    resolved_findings: list[ReportFinding]
    persistent_findings: list[ReportFinding]
    absent_findings: list[ReportFinding]
    uncertain_findings: list[ReportFinding]

    overall_change: str
    overall_uncertainty: str
    requires_review: bool

    impression: str
    review_notes: list[str]