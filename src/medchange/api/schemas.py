from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class APIErrorResponse(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ModelInfoResponse(BaseModel):
    biomedclip_model: str
    qwen_model: str
    qwen_quantization: str

    safety_policy: str
    safety_threshold: float

    target_findings: list[str]

class RuntimeStatusResponse(
    BaseModel
):
    busy: bool
    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    cache_entries: int
    active_request_id: str | None


class CacheClearResponse(
    BaseModel
):
    status: str
    cleared_entries: int


class FindingResponse(BaseModel):
    finding: str

    final_state: str

    biomedclip_state: str | None = None
    qwen_state: str | None = None

    biomedclip_confidence: float | None = None
    qwen_confidence: float | None = None

    agreement: str
    uncertainty: str

    requires_review: bool

    evidence: str | None = None
    decision_reason: str | None = None


class AnalyzePairResponse(BaseModel):
    pair_id: str

    prior_study_id: str
    current_study_id: str

    overall_change: str
    uncertainty: str
    requires_review: bool

    findings: list[FindingResponse]

    impression: str

    safety_policy: str
    safety_threshold: float

    total_elapsed_seconds: float
    cache_hit: bool = False