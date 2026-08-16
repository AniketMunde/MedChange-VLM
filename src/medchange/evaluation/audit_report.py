from __future__ import annotations

from medchange.evaluation.audit import (
    BenchmarkAudit,
)


def print_benchmark_audit(
    audit: BenchmarkAudit,
) -> None:
    print()
    print("=" * 110)
    print(
        "BENCHMARK AUDIT"
    )
    print("=" * 110)

    print(
        f"Requested samples       : "
        f"{audit.requested_samples}"
    )

    print(
        f"Actual samples          : "
        f"{audit.actual_samples}"
    )

    print(
        f"Development samples     : "
        f"{audit.development_samples}"
    )

    print(
        f"Test samples            : "
        f"{audit.test_samples}"
    )

    print(
        f"Development patients    : "
        f"{audit.development_unique_patients}"
    )

    print(
        f"Test patients           : "
        f"{audit.test_unique_patients}"
    )

    print(
        f"Patient overlap         : "
        f"{audit.patient_overlap_count}"
    )

    print(
        f"Seed                    : "
        f"{audit.seed}"
    )

    print(
        f"Development fraction    : "
        f"{audit.development_fraction:.3f}"
    )

    print()
    print(
        "Development views:"
    )

    for view, count in (
        audit.development_views.items()
    ):
        print(
            f"  {view:<10} {count}"
        )

    print()
    print(
        "Test views:"
    )

    for view, count in (
        audit.test_views.items()
    ):
        print(
            f"  {view:<10} {count}"
        )

    print()
    print("-" * 110)

    header = (
        f"{'Finding':<22}"
        f"{'DEV +':>8}"
        f"{'DEV Prev':>12}"
        f"{'TEST +':>10}"
        f"{'TEST Prev':>12}"
        f"{'Threshold':>12}"
    )

    print(
        header
    )

    print("-" * 110)

    for finding in audit.findings:
        threshold = (
            f"{finding.calibrated_threshold:.3f}"
            if finding.calibrated_threshold
            is not None
            else "N/A"
        )

        print(
            f"{finding.finding:<22}"
            f"{finding.development_positive:>8}"
            f"{finding.development_prevalence:>12.3%}"
            f"{finding.test_positive:>10}"
            f"{finding.test_prevalence:>12.3%}"
            f"{threshold:>12}"
        )

    print("-" * 110)

    warning_count = 0

    for finding in audit.findings:
        for warning in finding.warnings:
            warning_count += 1

            print(
                f"WARNING [{finding.finding}]: "
                f"{warning}"
            )

    for warning in audit.warnings:
        warning_count += 1

        print(
            f"WARNING [benchmark]: "
            f"{warning}"
        )

    if warning_count == 0:
        print(
            "No benchmark audit warnings."
        )

    print("=" * 110)