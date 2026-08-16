from __future__ import annotations

from medchange.reasoning.temporal_result import (
    FindingEvidence,
    UnifiedTemporalResult,
)
from medchange.reporting.report_schema import (
    LongitudinalReport,
    ReportFinding,
)


def _to_report_finding(
    finding: FindingEvidence,
) -> ReportFinding:
    return ReportFinding(
        finding=finding.finding,
        state=finding.final_state,
        agreement=finding.agreement,
        uncertainty=finding.uncertainty,
        evidence=finding.evidence,
        requires_review=finding.requires_review,
    )


def _pretty_finding(
    finding: str,
) -> str:
    return finding.replace(
        "_",
        " ",
    )


def _build_impression(
    result: UnifiedTemporalResult,
) -> str:
    new_findings = [
        _pretty_finding(
            item.finding
        )
        for item in result.findings
        if item.final_state == "new"
    ]

    resolved_findings = [
        _pretty_finding(
            item.finding
        )
        for item in result.findings
        if item.final_state == "resolved"
    ]

    persistent_findings = [
        _pretty_finding(
            item.finding
        )
        for item in result.findings
        if item.final_state == "persistent"
    ]

    uncertain_findings = [
        _pretty_finding(
            item.finding
        )
        for item in result.findings
        if item.final_state == "uncertain"
    ]

    parts = []

    if new_findings:
        parts.append(
            "New: "
            + ", ".join(
                new_findings
            )
            + "."
        )

    if resolved_findings:
        parts.append(
            "Resolved: "
            + ", ".join(
                resolved_findings
            )
            + "."
        )

    if persistent_findings:
        parts.append(
            "Persistent: "
            + ", ".join(
                persistent_findings
            )
            + "."
        )

    if uncertain_findings:
        parts.append(
            "Uncertain due to model disagreement or "
            "insufficient agreement: "
            + ", ".join(
                uncertain_findings
            )
            + "."
        )

    if not parts:
        parts.append(
            "No new, resolved, persistent, or uncertain "
            "target findings identified."
        )

    return " ".join(
        parts
    )


def _build_review_notes(
    result: UnifiedTemporalResult,
) -> list[str]:
    notes = []

    for item in result.findings:
        if not item.requires_review:
            continue

        bio_state = (
            item.biomedclip_state
            or "unavailable"
        )

        qwen_state = (
            item.qwen_state
            or "unavailable"
        )

        note = (
            f"{_pretty_finding(item.finding)}: "
            f"BiomedCLIP={bio_state}, "
            f"Qwen={qwen_state}, "
            f"agreement={item.agreement}, "
            f"uncertainty={item.uncertainty}."
        )

        if item.decision_reason:
            note += (
                " "
                + item.decision_reason
            )

        notes.append(
            note
        )

    return notes


def build_longitudinal_report(
    result: UnifiedTemporalResult,
) -> LongitudinalReport:
    """
    Convert a validated unified MedChange result into
    a deterministic longitudinal report.

    No generative model is used at this stage.
    """

    buckets = {
        "new": [],
        "resolved": [],
        "persistent": [],
        "absent": [],
        "uncertain": [],
    }

    for finding in result.findings:
        state = finding.final_state

        if state not in buckets:
            state = "uncertain"

        buckets[
            state
        ].append(
            _to_report_finding(
                finding
            )
        )

    impression = (
        _build_impression(
            result
        )
    )

    review_notes = (
        _build_review_notes(
            result
        )
    )

    return LongitudinalReport(
        pair_id=result.pair_id,
        prior_study_id=(
            result.prior_study_id
        ),
        current_study_id=(
            result.current_study_id
        ),

        new_findings=(
            buckets["new"]
        ),
        resolved_findings=(
            buckets["resolved"]
        ),
        persistent_findings=(
            buckets["persistent"]
        ),
        absent_findings=(
            buckets["absent"]
        ),
        uncertain_findings=(
            buckets["uncertain"]
        ),

        overall_change=(
            result.overall_change
        ),
        overall_uncertainty=(
            result.uncertainty
        ),
        requires_review=(
            result.requires_review
        ),

        impression=impression,
        review_notes=review_notes,
    )