from __future__ import annotations

from dataclasses import dataclass


NO_FINDING = "No Finding"


@dataclass(frozen=True)
class TemporalLabelChange:
    new: tuple[str, ...]
    resolved: tuple[str, ...]
    persistent: tuple[str, ...]

    @property
    def has_change(self) -> bool:
        return bool(
            self.new
            or self.resolved
        )


def _pathology_set(
    labels: tuple[str, ...],
) -> set[str]:
    return {
        label
        for label in labels
        if label != NO_FINDING
    }


def derive_temporal_label_change(
    prior_labels: tuple[str, ...],
    current_labels: tuple[str, ...],
) -> TemporalLabelChange:
    """
    Derive weak longitudinal supervision from
    NIH image-level pathology labels.

    Important:
    These labels indicate changes in annotated
    findings, not necessarily confirmed clinical
    progression or resolution.
    """

    prior = _pathology_set(
        prior_labels
    )

    current = _pathology_set(
        current_labels
    )

    new_findings = tuple(
        sorted(
            current - prior
        )
    )

    resolved_findings = tuple(
        sorted(
            prior - current
        )
    )

    persistent_findings = tuple(
        sorted(
            prior & current
        )
    )

    return TemporalLabelChange(
        new=new_findings,
        resolved=resolved_findings,
        persistent=persistent_findings,
    )