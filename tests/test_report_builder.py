from medchange.reasoning.temporal_result import (
    FindingEvidence,
    UnifiedTemporalResult,
)
from medchange.reporting.report_builder import (
    build_longitudinal_report,
)


def test_build_report_with_conflict():
    result = UnifiedTemporalResult(
        pair_id="8185_6_7",
        prior_study_id="prior",
        current_study_id="current",

        findings=[
            FindingEvidence(
                finding="atelectasis",
                final_state="uncertain",
                biomedclip_state="absent",
                qwen_state="new",
                biomedclip_confidence=0.75,
                qwen_confidence=0.82,
                agreement="conflict",
                uncertainty="high",
                evidence=(
                    "New opacity on current image."
                ),
                decision_reason=(
                    "BiomedCLIP and Qwen produced "
                    "conflicting temporal states."
                ),
                requires_review=True,
            ),

            FindingEvidence(
                finding="cardiomegaly",
                final_state="absent",
                biomedclip_state="absent",
                qwen_state="absent",
                biomedclip_confidence=0.99,
                qwen_confidence=0.95,
                agreement="high",
                uncertainty="low",
                evidence=(
                    "No interval cardiac enlargement."
                ),
                decision_reason=(
                    "BiomedCLIP and Qwen agree on "
                    "the temporal state."
                ),
                requires_review=False,
            ),
        ],

        overall_change="uncertain",
        uncertainty="high",
        requires_review=True,

        summary=(
            "Model disagreement requires review."
        ),
    )

    report = (
        build_longitudinal_report(
            result
        )
    )

    assert (
        len(
            report.uncertain_findings
        )
        == 1
    )

    assert (
        report.uncertain_findings[
            0
        ].finding
        == "atelectasis"
    )

    assert (
        report.requires_review
        is True
    )

    assert (
        len(
            report.review_notes
        )
        == 1
    )

    assert (
        "BiomedCLIP=absent"
        in report.review_notes[
            0
        ]
    )


def test_report_categories():
    result = UnifiedTemporalResult(
        pair_id="x",
        prior_study_id="prior",
        current_study_id="current",

        findings=[
            FindingEvidence(
                finding="pneumothorax",
                final_state="resolved",
                biomedclip_state="resolved",
                qwen_state="resolved",
                agreement="high",
                uncertainty="low",
                requires_review=False,
            ),

            FindingEvidence(
                finding="atelectasis",
                final_state="new",
                biomedclip_state="new",
                qwen_state="new",
                agreement="high",
                uncertainty="low",
                requires_review=False,
            ),

            FindingEvidence(
                finding="cardiomegaly",
                final_state="persistent",
                biomedclip_state="persistent",
                qwen_state="persistent",
                agreement="high",
                uncertainty="low",
                requires_review=False,
            ),
        ],

        overall_change="mixed",
        uncertainty="low",
        requires_review=False,
        summary="Mixed temporal changes.",
    )

    report = (
        build_longitudinal_report(
            result
        )
    )

    assert (
        len(
            report.new_findings
        )
        == 1
    )

    assert (
        len(
            report.resolved_findings
        )
        == 1
    )

    assert (
        len(
            report.persistent_findings
        )
        == 1
    )

    assert (
        report.overall_change
        == "mixed"
    )