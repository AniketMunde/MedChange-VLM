import pandas as pd

from medchange.evaluation.audit import (
    audit_benchmark,
)


def build_sample_data():
    development = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "2",
                "3",
                "4",
            ],

            "view_position": [
                "PA",
                "PA",
                "AP",
                "AP",
            ],

            "pleural_effusion_label": [
                1,
                0,
                1,
                0,
            ],

            "pneumonia_label": [
                0,
                0,
                1,
                0,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "patient_id": [
                "5",
                "6",
                "7",
                "8",
            ],

            "view_position": [
                "PA",
                "AP",
                "PA",
                "PA",
            ],

            "pleural_effusion_label": [
                0,
                1,
                0,
                1,
            ],

            "pneumonia_label": [
                0,
                0,
                0,
                1,
            ],
        }
    )

    thresholds = pd.DataFrame(
        {
            "finding": [
                "pleural_effusion",
                "pneumonia",
            ],

            "threshold": [
                0.55,
                0.95,
            ],
        }
    )

    return (
        development,
        test,
        thresholds,
    )


def test_patient_overlap_zero():
    (
        development,
        test,
        thresholds,
    ) = build_sample_data()

    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=[
            "pleural_effusion",
            "pneumonia",
        ],
        requested_samples=8,
        seed=42,
        development_fraction=0.5,
        min_positive_warning=1,
    )

    assert (
        audit.patient_overlap_count
        == 0
    )

    assert (
        audit.actual_samples
        == 8
    )


def test_extreme_threshold_warning():
    (
        development,
        test,
        thresholds,
    ) = build_sample_data()

    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=[
            "pleural_effusion",
            "pneumonia",
        ],
        requested_samples=8,
        seed=42,
        development_fraction=0.5,
        min_positive_warning=1,
    )

    pneumonia = next(
        finding
        for finding in audit.findings
        if finding.finding
        == "pneumonia"
    )

    assert any(
        "Very high calibrated threshold"
        in warning
        for warning
        in pneumonia.warnings
    )


def test_low_positive_warning():
    (
        development,
        test,
        thresholds,
    ) = build_sample_data()

    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=[
            "pleural_effusion",
            "pneumonia",
        ],
        requested_samples=8,
        seed=42,
        development_fraction=0.5,
        min_positive_warning=5,
    )

    pneumonia = next(
        finding
        for finding in audit.findings
        if finding.finding
        == "pneumonia"
    )

    assert any(
        "positive development examples"
        in warning
        for warning
        in pneumonia.warnings
    )

    assert any(
        "positive test examples"
        in warning
        for warning
        in pneumonia.warnings
    )


def test_view_distribution():
    (
        development,
        test,
        thresholds,
    ) = build_sample_data()

    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=[
            "pleural_effusion",
            "pneumonia",
        ],
        requested_samples=8,
        seed=42,
        development_fraction=0.5,
        min_positive_warning=1,
    )

    assert (
        audit.development_views[
            "PA"
        ]
        == 2
    )

    assert (
        audit.test_views[
            "PA"
        ]
        == 3
    )
def test_patient_leakage_detected():
    development = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "2",
            ],

            "view_position": [
                "PA",
                "PA",
            ],

            "pleural_effusion_label": [
                1,
                0,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "patient_id": [
                "1",
                "3",
            ],

            "view_position": [
                "PA",
                "AP",
            ],

            "pleural_effusion_label": [
                0,
                1,
            ],
        }
    )

    thresholds = pd.DataFrame(
        {
            "finding": [
                "pleural_effusion"
            ],

            "threshold": [
                0.5
            ],
        }
    )

    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=[
            "pleural_effusion"
        ],
        requested_samples=4,
        seed=42,
        development_fraction=0.5,
        min_positive_warning=1,
    )

    assert (
        audit.patient_overlap_count
        == 1
    )

    assert any(
        "Patient leakage detected"
        in warning
        for warning
        in audit.warnings
    )