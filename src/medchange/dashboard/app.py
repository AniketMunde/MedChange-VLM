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
    render_downloads,
    render_evidence_details,
    render_execution_metadata,
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

    pair_id = result.get(
        "pair_id",
        "N/A",
    )

    prior_study = result.get(
        "prior_study_id",
        "N/A",
    )

    current_study = result.get(
        "current_study_id",
        "N/A",
    )

    st.caption(
        f"Pair {pair_id} · "
        f"{prior_study} → "
        f"{current_study}"
    )

    # --------------------------------------------------------
    # HIGH-LEVEL STATUS
    # --------------------------------------------------------

    render_overview_metrics(
        result
    )

    # --------------------------------------------------------
    # FINDING TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Finding-level comparison"
    )

    findings = (
        result.get(
            "findings",
            [],
        )
    )

    if len(
        findings
    ) != 7:
        st.warning(
            "The response did not contain "
            "all seven validated findings."
        )

    render_findings_table(
        findings
    )

    # --------------------------------------------------------
    # IMPRESSION
    # --------------------------------------------------------

    st.divider()

    render_impression(
        result.get(
            "impression",
            "No longitudinal impression returned.",
        )
    )

    # --------------------------------------------------------
    # REVIEW FLAGS
    # --------------------------------------------------------

    render_review_flags(
        findings
    )

    # --------------------------------------------------------
    # DETAILED EVIDENCE
    # --------------------------------------------------------

    render_evidence_details(
        findings
    )

    # --------------------------------------------------------
    # RUNTIME / CONFIG
    # --------------------------------------------------------

    st.divider()

    render_execution_metadata(
        result
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    render_downloads(
        result
    )


def main() -> None:
    render_header()
    st.info(
        "Validated scope: longitudinal comparison of seven "
        "target findings — atelectasis, cardiomegaly, "
        "consolidation, edema, pleural effusion, pneumonia, "
        "and pneumothorax. Research prototype only."
    )

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
                    "Running MedChange analysis...",
                    expanded=True,
            ) as status:

                st.write(
                    "1. Validating longitudinal image pair"
                )

                st.write(
                    "2. Preparing safety-aware request"
                )

                st.write(
                    "3. Running temporal vision inference"
                )

                st.write(
                    "4. Running vision-language reasoning"
                )

                st.write(
                    "5. Reconciling model evidence"
                )

                st.write(
                    "6. Building longitudinal report"
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

                if result.get(
                        "cache_hit",
                        False,
                ):
                    status.update(
                        label=(
                            "Analysis restored from cache"
                        ),
                        state="complete",
                        expanded=False,
                    )

                else:
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