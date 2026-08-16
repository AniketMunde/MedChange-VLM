from pathlib import Path

import pytest
from PIL import Image

from medchange.data import (
    LongitudinalStudyDataset,
    StudyDataset,
)
from medchange.preprocessing import (
    MedicalImagePreprocessor,
)
from medchange.schemas import (
    LongitudinalStudyPair,
    Study,
)


def create_test_image(
    path: Path,
    size=(640, 800),
) -> None:
    image = Image.new(
        mode="L",
        size=size,
        color=128,
    )

    image.save(path)


def test_study_dataset(
    tmp_path: Path,
):
    image_path = (
        tmp_path / "study.png"
    )

    create_test_image(
        image_path
    )

    study = Study(
        patient_id="patient-001",
        study_id="study-001",
        image_path=image_path,
        report="No acute abnormality.",
    )

    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    dataset = StudyDataset(
        studies=[study],
        preprocessor=preprocessor,
    )

    sample = dataset[0]

    assert len(dataset) == 1

    assert (
        sample["patient_id"]
        == "patient-001"
    )

    assert (
        sample["pixel_values"].shape
        == (3, 448, 448)
    )


def test_longitudinal_dataset(
    tmp_path: Path,
):
    prior_path = (
        tmp_path / "prior.png"
    )

    current_path = (
        tmp_path / "current.png"
    )

    create_test_image(
        prior_path
    )

    create_test_image(
        current_path
    )

    prior = Study(
        patient_id="patient-001",
        study_id="study-prior",
        image_path=prior_path,
        report="Small right pleural effusion.",
    )

    current = Study(
        patient_id="patient-001",
        study_id="study-current",
        image_path=current_path,
        report=(
            "Increasing right pleural effusion."
        ),
    )

    pair = LongitudinalStudyPair(
        patient_id="patient-001",
        prior=prior,
        current=current,
        time_delta_days=5,
    )

    preprocessor = (
        MedicalImagePreprocessor(
            image_size=448
        )
    )

    dataset = (
        LongitudinalStudyDataset(
            pairs=[pair],
            preprocessor=preprocessor,
        )
    )

    sample = dataset[0]

    assert len(dataset) == 1

    assert (
        sample[
            "prior_pixel_values"
        ].shape
        == (3, 448, 448)
    )

    assert (
        sample[
            "current_pixel_values"
        ].shape
        == (3, 448, 448)
    )

    assert (
        sample["time_delta_days"]
        == pytest.approx(5)
    )


def test_longitudinal_dataset_rejects_mismatch(
    tmp_path: Path,
):
    prior_path = (
        tmp_path / "prior.png"
    )

    current_path = (
        tmp_path / "current.png"
    )

    create_test_image(
        prior_path
    )

    create_test_image(
        current_path
    )

    prior = Study(
        patient_id="patient-A",
        study_id="study-A",
        image_path=prior_path,
    )

    current = Study(
        patient_id="patient-B",
        study_id="study-B",
        image_path=current_path,
    )

    pair = LongitudinalStudyPair(
        patient_id="patient-A",
        prior=prior,
        current=current,
    )

    preprocessor = (
        MedicalImagePreprocessor()
    )

    with pytest.raises(
        ValueError
    ):
        LongitudinalStudyDataset(
            pairs=[pair],
            preprocessor=preprocessor,
        )