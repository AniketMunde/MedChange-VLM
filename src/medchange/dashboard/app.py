from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from medchange.dashboard.api_client import (
    APIConfig,
    MedChangeAPIClient,
    MedChangeAPIError,
)
from medchange.dashboard.components import (
    render_api_status,
    render_evidence_details,
    render_findings_table,
    render_header,
    render_impression,
    render_overview_metrics,
    render_review_flags,
)
from medchange.dashboard.styling import (
    dashboard_css,
)


st.set_page_config(
    page_title="MedChange-VLM",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    dashboard_css(),
    unsafe_allow_html=True,
)


def save_uploaded_file(
    uploaded_file,
    directory: Path,
    name: str,
) -> Path:
    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".png"
    )

    path = (
        directory
        / f"{name}{suffix}"
    )

    path.write_bytes(
        uploaded_file.getbuffer()
    )

    return path


def build_client() -> MedChangeAPIClient:
    api_url = st.session_state.get(
        "api_url",
        "http://127.0.0.1:8000",
    )

    return MedChangeAPIClient(
        APIConfig(
            base_url=api_url,
            timeout_seconds=240,
        )
    )


def render_sidebar() -> tuple[
    str,
    float,
]:
    with st.sidebar:
        st.markdown(
            "## Configuration"
        )

        st.caption(
            "Inference and safety settings"
        )

        st.text_input(
            "API URL",
            value=(
                "http://127.0.0.1:8000"
            ),
            key="api_url",
        )

        st.divider()

        safety_policy = st.selectbox(
            "Safety policy",
            options=[
                "change_sensitive",
                "strict",
            ],
            index=0,
            help=(
                "change_sensitive preserves "
                "BiomedCLIP change-state predictions "
                "while flagging disagreement. "
                "strict abstains on model conflict."
            ),
        )

        default_threshold = (
            0.80
            if safety_policy
            == "change_sensitive"
            else 0.60
        )

        safety_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                default_threshold
            ),
            step=0.05,
        )

        st.divider()

        st.markdown(
            "### System"
        )

        client = (
            build_client()
        )

        try:
            health = (
                client.health()
            )

            render_api_status(
                True,
                health.get(
                    "status",
                    "ok",
                ),
            )

        except MedChangeAPIError as exc:
            render_api_status(
                False,
                str(
                    exc
                ),
            )

        with st.expander(
            "Model configuration"
        ):
            try:
                info = (
                    client.model_info()
                )

                st.markdown(
                    "**Vision encoder**"
                )
                st.caption(
                    info.get(
                        "biomedclip_model",
                        "Unknown",
                    )
                )

                st.markdown(
                    "**Vision-language model**"
                )
                st.caption(
                    info.get(
                        "qwen_model",
                        "Unknown",
                    )
                )

                st.markdown(
                    "**Quantization**"
                )
                st.caption(
                    info.get(
                        "qwen_quantization",
                        "Unknown",
                    )
                )

            except MedChangeAPIError:
                st.caption(
                    "Model information unavailable."
                )

        st.divider()

        st.caption(
            "Research prototype. "
            "Not intended for clinical diagnosis."
        )

    return (
        safety_policy,
        safety_threshold,
    )


def render_upload_panel():
    st.markdown(
        "## Study comparison"
    )

    st.caption(
        "Upload a prior and a current chest radiograph "
        "from the same longitudinal case."
    )

    image_columns = st.columns(
        2,
        gap="large",
    )

    with image_columns[0]:
        st.markdown(
            "### Prior study"
        )

        prior_file = (
            st.file_uploader(
                "Upload prior chest X-ray",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                ],
                key="prior_image",
            )
        )

        prior_study_id = (
            st.text_input(
                "Prior study ID",
                value="prior-study",
            )
        )

        if prior_file:
            st.image(
                prior_file,
                caption="Prior image",
                width="stretch",
            )

    with image_columns[1]:
        st.markdown(
            "### Current study"
        )

        current_file = (
            st.file_uploader(
                "Upload current chest X-ray",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                ],
                key="current_image",
            )
        )

        current_study_id = (
            st.text_input(
                "Current study ID",
                value="current-study",
            )
        )

        if current_file:
            st.image(
                current_file,
                caption="Current image",
                width="stretch",
            )

    pair_id = st.text_input(
        "Pair ID",
        value="medchange-case-001",
    )

    return (
        prior_file,
        current_file,
        prior_study_id,
        current_study_id,
        pair_id,
    )


def run_analysis(
    *,
    prior_file,
    current_file,
    prior_study_id: str,
    current_study_id: str,
    pair_id: str,
    safety_policy: str,
    safety_threshold: float,
):
    with tempfile.TemporaryDirectory(
        prefix="medchange_dashboard_"
    ) as temp_dir:
        temp_dir = Path(
            temp_dir
        )

        prior_path = (
            save_uploaded_file(
                prior_file,
                temp_dir,
                "prior",
            )
        )

        current_path = (
            save_uploaded_file(
                current_file,
                temp_dir,
                "current",
            )
        )

        client = (
            build_client()
        )

        return client.analyze_pair(
            prior_path=(
                prior_path
            ),
            current_path=(
                current_path
            ),
            pair_id=(
                pair_id
            ),
            prior_study_id=(
                prior_study_id
            ),
            current_study_id=(
                current_study_id
            ),
            safety_policy=(
                safety_policy
            ),
            safety_threshold=(
                safety_threshold
            ),
        )


def render_results(
    result: dict,
) -> None:
    st.divider()

    st.markdown(
        "## Longitudinal analysis"
    )

    st.caption(
        f"Pair ID: "
        f"{result.get('pair_id', 'N/A')}"
    )

    render_overview_metrics(
        result
    )

    st.markdown(
        "### Finding-level comparison"
    )

    findings = (
        result.get(
            "findings",
            [],
        )
    )

    render_findings_table(
        findings
    )

    st.divider()

    render_impression(
        result.get(
            "impression",
            "No impression returned.",
        )
    )

    render_review_flags(
        findings
    )

    render_evidence_details(
        findings
    )

    with st.expander(
        "Run metadata",
        expanded=False,
    ):
        metadata_columns = (
            st.columns(
                3
            )
        )

        metadata_columns[
            0
        ].metric(
            "Safety policy",
            result.get(
                "safety_policy",
                "N/A",
            ),
        )

        metadata_columns[
            1
        ].metric(
            "Threshold",
            f"{float(result.get('safety_threshold', 0)):.2f}",
        )

        metadata_columns[
            2
        ].metric(
            "Runtime",
            (
                f"{float(result.get('total_elapsed_seconds', 0)):.1f}s"
            ),
        )


def main() -> None:
    render_header()

    (
        safety_policy,
        safety_threshold,
    ) = render_sidebar()

    (
        prior_file,
        current_file,
        prior_study_id,
        current_study_id,
        pair_id,
    ) = render_upload_panel()

    st.markdown("")

    ready = (
        prior_file is not None
        and current_file is not None
        and bool(
            pair_id.strip()
        )
        and bool(
            prior_study_id.strip()
        )
        and bool(
            current_study_id.strip()
        )
    )

    analyze = st.button(
        "Analyze longitudinal change",
        type="primary",
        width="stretch",
        disabled=not ready,
    )

    if analyze:
        try:
            with st.status(
                "Running longitudinal analysis...",
                expanded=True,
            ) as status:

                st.write(
                    "Validating uploaded image pair..."
                )

                st.write(
                    "Running BiomedCLIP temporal inference..."
                )

                st.write(
                    "Running Qwen2.5-VL comparison..."
                )

                st.write(
                    "Applying safety-aware decision policy..."
                )

                result = run_analysis(
                    prior_file=(
                        prior_file
                    ),
                    current_file=(
                        current_file
                    ),
                    prior_study_id=(
                        prior_study_id
                    ),
                    current_study_id=(
                        current_study_id
                    ),
                    pair_id=(
                        pair_id
                    ),
                    safety_policy=(
                        safety_policy
                    ),
                    safety_threshold=(
                        safety_threshold
                    ),
                )

                status.update(
                    label=(
                        "Analysis complete"
                    ),
                    state="complete",
                    expanded=False,
                )

            st.session_state[
                "medchange_result"
            ] = result

        except MedChangeAPIError as exc:
            st.error(
                str(
                    exc
                )
            )

        except Exception as exc:
            st.exception(
                exc
            )

    if (
        "medchange_result"
        in st.session_state
    ):
        render_results(
            st.session_state[
                "medchange_result"
            ]
        )

    st.markdown(
        """
        <div class="footer-note">
            MedChange-VLM is a research prototype for
            longitudinal chest X-ray analysis.
            Outputs should not be interpreted as clinical
            diagnoses or used independently for patient care.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()