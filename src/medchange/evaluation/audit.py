from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FindingAudit:
    finding: str

    development_positive: int
    development_negative: int
    development_prevalence: float

    test_positive: int
    test_negative: int
    test_prevalence: float

    calibrated_threshold: float | None

    warnings: list[str]


@dataclass(frozen=True)
class BenchmarkAudit:
    requested_samples: int

    actual_samples: int

    development_samples: int
    test_samples: int

    development_unique_patients: int
    test_unique_patients: int

    patient_overlap_count: int

    seed: int

    development_fraction: float

    development_views: dict[str, int]
    test_views: dict[str, int]

    findings: list[FindingAudit]

    warnings: list[str]


def _view_distribution(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    if "view_position" not in dataframe.columns:
        return {}

    counts = (
        dataframe[
            "view_position"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    return {
        str(view): int(count)
        for view, count
        in counts.items()
    }


def _label_statistics(
    dataframe: pd.DataFrame,
    finding: str,
) -> tuple[
    int,
    int,
    float,
]:
    column = (
        f"{finding}_label"
    )

    if column not in dataframe.columns:
        raise ValueError(
            f"Missing label column: {column}"
        )

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    valid = values[
        values.isin(
            [0, 1]
        )
    ]

    positive = int(
        (valid == 1).sum()
    )

    negative = int(
        (valid == 0).sum()
    )

    total = (
        positive
        + negative
    )

    prevalence = (
        positive / total
        if total > 0
        else 0.0
    )

    return (
        positive,
        negative,
        prevalence,
    )


def _get_threshold(
    threshold_table: pd.DataFrame,
    finding: str,
) -> float | None:
    if threshold_table.empty:
        return None

    required = {
        "finding",
        "threshold",
    }

    if not required.issubset(
        threshold_table.columns
    ):
        return None

    match = threshold_table[
        threshold_table[
            "finding"
        ]
        == finding
    ]

    if match.empty:
        return None

    return float(
        match.iloc[0][
            "threshold"
        ]
    )


def audit_benchmark(
    development: pd.DataFrame,
    test: pd.DataFrame,
    threshold_table: pd.DataFrame,
    findings: list[str],
    requested_samples: int,
    seed: int,
    development_fraction: float,
    min_positive_warning: int = 20,
    extreme_threshold_low: float = 0.10,
    extreme_threshold_high: float = 0.90,
) -> BenchmarkAudit:
    """
    Audit a biomedical benchmark before interpreting results.

    Checks:
    - patient leakage
    - actual sample count
    - class prevalence
    - low-positive findings
    - extreme calibrated thresholds
    - AP/PA distribution
    - reproducibility metadata
    """

    if development.empty:
        raise ValueError(
            "Development dataframe is empty."
        )

    if test.empty:
        raise ValueError(
            "Test dataframe is empty."
        )

    if (
        "patient_id"
        not in development.columns
        or "patient_id"
        not in test.columns
    ):
        raise ValueError(
            "Both subsets must contain patient_id."
        )

    development_patients = set(
        development[
            "patient_id"
        ]
        .astype(str)
    )

    test_patients = set(
        test[
            "patient_id"
        ]
        .astype(str)
    )

    overlap = (
        development_patients
        & test_patients
    )

    global_warnings: list[str] = []

    if overlap:
        global_warnings.append(
            "Patient leakage detected between "
            "development and test subsets."
        )

    actual_samples = (
        len(development)
        + len(test)
    )

    if (
        actual_samples
        != requested_samples
    ):
        global_warnings.append(
            "Actual sample count differs from "
            f"requested sample count: "
            f"{actual_samples} vs "
            f"{requested_samples}."
        )

    finding_audits: list[
        FindingAudit
    ] = []

    for finding in findings:
        (
            dev_positive,
            dev_negative,
            dev_prevalence,
        ) = _label_statistics(
            development,
            finding,
        )

        (
            test_positive,
            test_negative,
            test_prevalence,
        ) = _label_statistics(
            test,
            finding,
        )

        threshold = _get_threshold(
            threshold_table,
            finding,
        )

        finding_warnings: list[
            str
        ] = []

        if (
            dev_positive
            < min_positive_warning
        ):
            finding_warnings.append(
                f"Only {dev_positive} positive "
                "development examples; threshold "
                "calibration may be unstable."
            )

        if (
            test_positive
            < min_positive_warning
        ):
            finding_warnings.append(
                f"Only {test_positive} positive "
                "test examples; metrics and "
                "confidence intervals may be unstable."
            )

        if threshold is not None:
            if (
                threshold
                <= extreme_threshold_low
            ):
                finding_warnings.append(
                    f"Very low calibrated threshold: "
                    f"{threshold:.3f}."
                )

            if (
                threshold
                >= extreme_threshold_high
            ):
                finding_warnings.append(
                    f"Very high calibrated threshold: "
                    f"{threshold:.3f}."
                )

        finding_audits.append(
            FindingAudit(
                finding=finding,

                development_positive=(
                    dev_positive
                ),

                development_negative=(
                    dev_negative
                ),

                development_prevalence=(
                    dev_prevalence
                ),

                test_positive=(
                    test_positive
                ),

                test_negative=(
                    test_negative
                ),

                test_prevalence=(
                    test_prevalence
                ),

                calibrated_threshold=(
                    threshold
                ),

                warnings=(
                    finding_warnings
                ),
            )
        )

    return BenchmarkAudit(
        requested_samples=(
            requested_samples
        ),

        actual_samples=(
            actual_samples
        ),

        development_samples=(
            len(development)
        ),

        test_samples=(
            len(test)
        ),

        development_unique_patients=(
            len(
                development_patients
            )
        ),

        test_unique_patients=(
            len(
                test_patients
            )
        ),

        patient_overlap_count=(
            len(overlap)
        ),

        seed=seed,

        development_fraction=(
            development_fraction
        ),

        development_views=(
            _view_distribution(
                development
            )
        ),

        test_views=(
            _view_distribution(
                test
            )
        ),

        findings=(
            finding_audits
        ),

        warnings=(
            global_warnings
        ),
    )


def benchmark_audit_to_dict(
    audit: BenchmarkAudit,
) -> dict[str, Any]:
    return asdict(
        audit
    )