from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

from torch.utils.data import Dataset

from medchange.preprocessing import (
    MedicalImagePreprocessor,
)
from medchange.schemas import (
    LongitudinalStudyPair,
    Study,
)


class StudyDataset(Dataset):
    """
    Dataset for independent radiographic studies.
    """

    def __init__(
        self,
        studies: Sequence[Study],
        preprocessor: MedicalImagePreprocessor,
    ) -> None:
        self.studies = list(studies)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        study = self.studies[index]

        processed = self.preprocessor(
            study.image_path
        )

        return {
            "patient_id": study.patient_id,
            "study_id": study.study_id,
            "pixel_values": processed.pixel_values,
            "report": study.report,
            "study_datetime": study.study_datetime,
            "image_metadata": processed.metadata,
        }


class LongitudinalStudyDataset(Dataset):
    """
    Dataset containing prior/current radiographic study pairs.
    """

    def __init__(
        self,
        pairs: Sequence[LongitudinalStudyPair],
        preprocessor: MedicalImagePreprocessor,
    ) -> None:
        self.pairs = list(pairs)
        self.preprocessor = preprocessor

        for pair in self.pairs:
            pair.validate_same_patient()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        pair = self.pairs[index]

        prior = self.preprocessor(
            pair.prior.image_path
        )

        current = self.preprocessor(
            pair.current.image_path
        )

        return {
            "patient_id": pair.patient_id,

            "prior_study_id":
                pair.prior.study_id,

            "current_study_id":
                pair.current.study_id,

            "prior_pixel_values":
                prior.pixel_values,

            "current_pixel_values":
                current.pixel_values,

            "prior_report":
                pair.prior.report,

            "current_report":
                pair.current.report,

            "prior_metadata":
                prior.metadata,

            "current_metadata":
                current.metadata,

            "time_delta_days":
                pair.time_delta_days,
        }