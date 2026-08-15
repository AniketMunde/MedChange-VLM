from pathlib import Path

import pytest

from medchange.schemas import (
    BoundingBox,
    LongitudinalStudyPair,
    MedicalFinding,
    MedChangePrediction,
    Study,
    TemporalStatus,
)


def test_create_study():
    study = Study(
        patient_id="patient-001",
        study_id="study-001",
        image_path=Path(
            "data/samples/example.png"
        ),
    )

    assert study.patient_id == "patient-001"
    assert study.study_id == "study-001"


def test_medical_finding():
    finding = MedicalFinding(
        name="pleural_effusion",
        present=True,
        confidence=0.91,
        temporal_status=TemporalStatus.WORSENED,
    )

    assert finding.present is True
    assert finding.confidence == pytest.approx(0.91)
    assert (
        finding.temporal_status
        == TemporalStatus.WORSENED
    )


def test_prediction():
    finding = MedicalFinding(
        name="pleural_effusion",
        present=True,
        confidence=0.89,
        temporal_status=TemporalStatus.NEW,
    )

    prediction = MedChangePrediction(
        study_id="study-002",
        findings=[finding],
        comparison=(
            "New right-sided pleural effusion."
        ),
    )

    assert len(prediction.findings) == 1

    assert (
        prediction.findings[0].name
        == "pleural_effusion"
    )


def test_bounding_box():
    box = BoundingBox(
        x_min=0.10,
        y_min=0.20,
        x_max=0.80,
        y_max=0.90,
    )

    box.validate_geometry()

    assert box.x_min == pytest.approx(0.10)


def test_longitudinal_pair():
    prior = Study(
        patient_id="patient-001",
        study_id="study-prior",
        image_path=Path(
            "data/samples/prior.png"
        ),
    )

    current = Study(
        patient_id="patient-001",
        study_id="study-current",
        image_path=Path(
            "data/samples/current.png"
        ),
    )

    pair = LongitudinalStudyPair(
        patient_id="patient-001",
        prior=prior,
        current=current,
        time_delta_days=5,
    )

    pair.validate_same_patient()

    assert pair.time_delta_days == pytest.approx(5)