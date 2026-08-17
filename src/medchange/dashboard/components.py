from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import json


def render_header() -> None:
    st.markdown(
        """
        <div class="medchange-header">
            <div class="medchange-title">
                MedChange-VLM
            </div>
            <div class="medchange-subtitle">
                Safety-aware longitudinal chest X-ray comparison
                using BiomedCLIP temporal representations and
                Qwen2.5-VL reasoning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_status(
    connected: bool,
    detail: str,
) -> None:
    if connected:
        st.success(
            f"API connected · {detail}",
            icon="✅",
        )
    else:
        st.error(
            f"API unavailable · {detail}",
            icon="⚠️",
        )


def render_metric_card(
    *,
    label: str,
    value: str,
    caption: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
            <div class="metric-caption">
                {caption}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_metrics(
    result: dict[str, Any],
) -> None:
    columns = st.columns(
        4
    )

    with columns[0]:
        render_metric_card(
            label="Overall change",
            value=str(
                result.get(
                    "overall_change",
                    "N/A",
                )
            ).upper(),
            caption="Final longitudinal assessment",
        )

    with columns[1]:
        render_metric_card(
            label="Uncertainty",
            value=str(
                result.get(
                    "uncertainty",
                    "N/A",
                )
            ).upper(),
            caption="Decision-layer uncertainty",
        )

    with columns[2]:
        review = bool(
            result.get(
                "requires_review",
                False,
            )
        )

        render_metric_card(
            label="Review",
            value=(
                "REQUIRED"
                if review
                else "NOT REQUIRED"
            ),
            caption=(
                "Manual review recommended"
                if review
                else "No review flag raised"
            ),
        )

    with columns[3]:
        elapsed = result.get(
            "total_elapsed_seconds"
        )

        render_metric_card(
            label="Inference time",
            value=(
                f"{float(elapsed):.1f} s"
                if elapsed is not None
                else "N/A"
            ),
            caption="End-to-end API runtime",
        )


def build_findings_dataframe(
    findings: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for item in findings:
        rows.append(
            {
                "Finding": (
                    str(
                        item.get(
                            "finding",
                            "",
                        )
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                ),

                "Final": (
                    item.get(
                        "final_state"
                    )
                ),

                "BiomedCLIP": (
                    item.get(
                        "biomedclip_state"
                    )
                ),

                "Bio confidence": (
                    item.get(
                        "biomedclip_confidence"
                    )
                ),

                "Qwen": (
                    item.get(
                        "qwen_state"
                    )
                ),

                "Qwen confidence": (
                    item.get(
                        "qwen_confidence"
                    )
                ),

                "Agreement": (
                    item.get(
                        "agreement"
                    )
                ),

                "Uncertainty": (
                    item.get(
                        "uncertainty"
                    )
                ),

                "Review": (
                    item.get(
                        "requires_review"
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def render_findings_table(
    findings: list[dict[str, Any]],
) -> None:
    dataframe = (
        build_findings_dataframe(
            findings
        )
    )

    st.dataframe(
        dataframe,
        width="stretch",
        hide_index=True,
        column_config={
            "Bio confidence":
                st.column_config.NumberColumn(
                    format="%.3f",
                ),

            "Qwen confidence":
                st.column_config.NumberColumn(
                    format="%.3f",
                ),

            "Review":
                st.column_config.CheckboxColumn(),
        },
    )


def render_impression(
    impression: str,
) -> None:
    st.markdown(
        """
        ### Longitudinal impression
        """
    )

    st.markdown(
        f"""
        <div class="report-box">
            {impression}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_flags(
    findings: list[dict[str, Any]],
) -> None:
    flagged = [
        finding
        for finding in findings
        if finding.get(
            "requires_review",
            False,
        )
    ]

    st.markdown(
        "### Review flags"
    )

    if not flagged:
        st.info(
            "No finding-level review flags were raised."
        )

        return

    for finding in flagged:
        name = (
            str(
                finding.get(
                    "finding",
                    "",
                )
            )
            .replace(
                "_",
                " ",
            )
            .title()
        )

        evidence = finding.get(
            "evidence"
        ) or "No explanatory evidence available."

        reason = finding.get(
            "decision_reason"
        ) or "No decision reason available."

        st.markdown(
            f"""
            <div class="review-box">
                <div class="finding-title">
                    {name}
                </div>
                <br>
                <strong>Model states:</strong>
                BiomedCLIP =
                {finding.get("biomedclip_state")}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Qwen =
                {finding.get("qwen_state")}
                <br><br>
                <strong>Evidence:</strong>
                {evidence}
                <br><br>
                <strong>Decision trace:</strong>
                {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_details(
    findings: list[dict[str, Any]],
) -> None:
    with st.expander(
        "Detailed model evidence",
        expanded=False,
    ):
        for finding in findings:
            name = (
                str(
                    finding.get(
                        "finding",
                        "",
                    )
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

            st.markdown(
                f"#### {name}"
            )

            columns = st.columns(
                3
            )

            with columns[0]:
                st.markdown(
                    "**Final state**"
                )

                st.write(
                    finding.get(
                        "final_state"
                    )
                )

            with columns[1]:
                st.markdown(
                    "**BiomedCLIP**"
                )

                st.write(
                    finding.get(
                        "biomedclip_state"
                    )
                )

                confidence = finding.get(
                    "biomedclip_confidence"
                )

                if confidence is not None:
                    st.caption(
                        f"Confidence: "
                        f"{confidence:.3f}"
                    )

            with columns[2]:
                st.markdown(
                    "**Qwen**"
                )

                st.write(
                    finding.get(
                        "qwen_state"
                    )
                )

                confidence = finding.get(
                    "qwen_confidence"
                )

                if confidence is not None:
                    st.caption(
                        f"Confidence: "
                        f"{confidence:.3f}"
                    )

            if finding.get(
                "evidence"
            ):
                st.markdown(
                    "**Qwen evidence**"
                )

                st.write(
                    finding[
                        "evidence"
                    ]
                )

            if finding.get(
                "decision_reason"
            ):
                st.markdown(
                    "**Decision trace**"
                )

                st.write(
                    finding[
                        "decision_reason"
                    ]
                )

            st.divider()
def build_text_report(
    result: dict[str, Any],
) -> str:
    lines = []

    lines.append(
        "MedChange-VLM — Longitudinal Chest X-ray Analysis"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        f"Pair ID: {result.get('pair_id', 'N/A')}"
    )

    lines.append(
        f"Prior study: "
        f"{result.get('prior_study_id', 'N/A')}"
    )

    lines.append(
        f"Current study: "
        f"{result.get('current_study_id', 'N/A')}"
    )

    lines.append("")

    lines.append(
        f"Overall change: "
        f"{result.get('overall_change', 'N/A')}"
    )

    lines.append(
        f"Uncertainty: "
        f"{result.get('uncertainty', 'N/A')}"
    )

    lines.append(
        f"Requires review: "
        f"{result.get('requires_review', False)}"
    )

    lines.append("")

    lines.append(
        "LONGITUDINAL IMPRESSION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        str(
            result.get(
                "impression",
                "No impression available.",
            )
        )
    )

    lines.append("")

    lines.append(
        "FINDING-LEVEL RESULTS"
    )

    lines.append(
        "-" * 72
    )

    findings = result.get(
        "findings",
        [],
    )

    for item in findings:
        finding_name = (
            str(
                item.get(
                    "finding",
                    "",
                )
            )
            .replace(
                "_",
                " ",
            )
            .title()
        )

        lines.append(
            finding_name
        )

        lines.append(
            f"  Final state: "
            f"{item.get('final_state')}"
        )

        lines.append(
            f"  BiomedCLIP: "
            f"{item.get('biomedclip_state')}"
        )

        lines.append(
            f"  Qwen: "
            f"{item.get('qwen_state')}"
        )

        lines.append(
            f"  Agreement: "
            f"{item.get('agreement')}"
        )

        lines.append(
            f"  Uncertainty: "
            f"{item.get('uncertainty')}"
        )

        lines.append(
            f"  Review: "
            f"{item.get('requires_review')}"
        )

        evidence = item.get(
            "evidence"
        )

        if evidence:
            lines.append(
                f"  Evidence: "
                f"{evidence}"
            )

        reason = item.get(
            "decision_reason"
        )

        if reason:
            lines.append(
                f"  Decision trace: "
                f"{reason}"
            )

        lines.append("")

    lines.append(
        "RUN CONFIGURATION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"Safety policy: "
        f"{result.get('safety_policy', 'N/A')}"
    )

    lines.append(
        f"Safety threshold: "
        f"{result.get('safety_threshold', 'N/A')}"
    )

    lines.append(
        f"Runtime: "
        f"{result.get('total_elapsed_seconds', 'N/A')} seconds"
    )

    lines.append(
        f"Cache hit: "
        f"{result.get('cache_hit', False)}"
    )

    lines.append("")

    lines.append(
        "Research prototype only. "
        "Not intended for clinical diagnosis "
        "or independent patient-care decisions."
    )

    return "\n".join(
        lines
    )


def render_downloads(
    result: dict[str, Any],
) -> None:
    st.markdown(
        "### Export"
    )

    columns = st.columns(
        2
    )

    json_payload = json.dumps(
        result,
        indent=2,
    )

    report_text = (
        build_text_report(
            result
        )
    )

    pair_id = str(
        result.get(
            "pair_id",
            "medchange-result",
        )
    )

    with columns[0]:
        st.download_button(
            label=(
                "Download structured JSON"
            ),

            data=(
                json_payload
            ),

            file_name=(
                f"{pair_id}_medchange.json"
            ),

            mime=(
                "application/json"
            ),

            width="stretch",
        )

    with columns[1]:
        st.download_button(
            label=(
                "Download longitudinal report"
            ),

            data=(
                report_text
            ),

            file_name=(
                f"{pair_id}_report.txt"
            ),

            mime=(
                "text/plain"
            ),

            width="stretch",
        )
def render_execution_metadata(
    result: dict[str, Any],
) -> None:
    st.markdown(
        "### Execution details"
    )

    columns = st.columns(
        4
    )

    with columns[0]:
        st.metric(
            "Safety policy",
            str(
                result.get(
                    "safety_policy",
                    "N/A",
                )
            ),
        )

    with columns[1]:
        threshold = result.get(
            "safety_threshold"
        )

        st.metric(
            "Threshold",
            (
                f"{float(threshold):.2f}"
                if threshold is not None
                else "N/A"
            ),
        )

    with columns[2]:
        elapsed = result.get(
            "total_elapsed_seconds"
        )

        st.metric(
            "Runtime",
            (
                f"{float(elapsed):.1f}s"
                if elapsed is not None
                else "N/A"
            ),
        )

    with columns[3]:
        cache_hit = bool(
            result.get(
                "cache_hit",
                False,
            )
        )

        st.metric(
            "Response source",
            (
                "Cache"
                if cache_hit
                else "Inference"
            ),
        )