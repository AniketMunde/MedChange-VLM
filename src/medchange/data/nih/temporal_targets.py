from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TemporalTarget(str, Enum):
    NEW = "new"
    RESOLVED = "resolved"
    PERSISTENT = "persistent"
    ABSENT = "absent"


NIH_TEMPORAL_FINDINGS = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}


@dataclass(frozen=True)
class FindingTemporalTarget:
    finding: str
    state: TemporalTarget


def _pathology_set(
    labels: tuple[str, ...],
) -> set[str]:
    return {
        label
        for label in labels
        if label != "No Finding"
    }


def derive_finding_temporal_target(
    finding: str,
    prior_labels: tuple[str, ...],
    current_labels: tuple[str, ...],
) -> TemporalTarget:
    """
    Derive a temporal state for one NIH pathology.

    NEW:
        absent in prior, present in current

    RESOLVED:
        present in prior, absent in current

    PERSISTENT:
        present in both

    ABSENT:
        absent in both
    """

    if finding not in NIH_TEMPORAL_FINDINGS:
        raise ValueError(
            f"Unsupported NIH temporal finding: {finding}"
        )

    prior = _pathology_set(
        prior_labels
    )

    current = _pathology_set(
        current_labels
    )

    prior_present = (
        finding in prior
    )

    current_present = (
        finding in current
    )

    if (
        not prior_present
        and current_present
    ):
        return TemporalTarget.NEW

    if (
        prior_present
        and not current_present
    ):
        return TemporalTarget.RESOLVED

    if (
        prior_present
        and current_present
    ):
        return TemporalTarget.PERSISTENT

    return TemporalTarget.ABSENT


def derive_all_temporal_targets(
    prior_labels: tuple[str, ...],
    current_labels: tuple[str, ...],
) -> dict[str, TemporalTarget]:
    """
    Derive temporal targets for all MedChange pathologies.
    """

    targets: dict[
        str,
        TemporalTarget,
    ] = {}

    for (
        nih_finding,
        medchange_finding,
    ) in NIH_TEMPORAL_FINDINGS.items():

        targets[
            medchange_finding
        ] = derive_finding_temporal_target(
            finding=nih_finding,
            prior_labels=prior_labels,
            current_labels=current_labels,
        )

    return targets