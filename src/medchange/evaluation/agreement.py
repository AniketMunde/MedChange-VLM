from __future__ import annotations

from dataclasses import dataclass

from medchange.schemas import (
    MedChangePrediction,
)


@dataclass
class FindingAgreement:
    finding: str

    qwen_present: bool | None

    qwen_confidence: float | None

    biomedclip_score: float

    agreement: str


def normalize_finding_name(
    finding: str,
) -> str:
    return (
        finding.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def compare_qwen_biomedclip(
    qwen_prediction: MedChangePrediction,
    biomedclip_scores: dict[str, float],
    threshold: float = 0.5,
) -> list[FindingAgreement]:
    """
    Describe agreement between Qwen-generated findings
    and BiomedCLIP scores.

    This does NOT perform clinical evidence fusion.
    """

    qwen_findings = {
        normalize_finding_name(
            finding.name
        ): finding
        for finding
        in qwen_prediction.findings
    }

    results = []

    for finding, score in (
        biomedclip_scores.items()
    ):

        normalized = (
            normalize_finding_name(
                finding
            )
        )

        qwen_finding = (
            qwen_findings.get(
                normalized
            )
        )

        biomedclip_present = (
            score >= threshold
        )

        if qwen_finding is None:
            qwen_present = False
            qwen_confidence = None

        else:
            qwen_present = (
                qwen_finding.present
            )

            qwen_confidence = (
                qwen_finding.confidence
            )

        if (
            qwen_present
            == biomedclip_present
        ):
            agreement = "agree"

        else:
            agreement = "disagree"

        results.append(
            FindingAgreement(
                finding=normalized,

                qwen_present=(
                    qwen_present
                ),

                qwen_confidence=(
                    qwen_confidence
                ),

                biomedclip_score=(
                    score
                ),

                agreement=agreement,
            )
        )

    return results