from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NIHLongitudinalStudy:
    image_index: str
    patient_id: str
    follow_up_number: int
    view_position: str | None
    labels: tuple[str, ...]


@dataclass(frozen=True)
class NIHLongitudinalPair:
    patient_id: str

    prior_image_index: str
    current_image_index: str

    prior_follow_up: int
    current_follow_up: int

    prior_labels: tuple[str, ...]
    current_labels: tuple[str, ...]

    prior_view: str | None
    current_view: str | None

    follow_up_delta: int

    @property
    def pair_id(self) -> str:
        return (
            f"{self.patient_id}_"
            f"{self.prior_follow_up}_"
            f"{self.current_follow_up}"
        )

    @property
    def same_view(self) -> bool:
        return (
            self.prior_view
            == self.current_view
        )


def normalize_labels(
    value: object,
) -> tuple[str, ...]:
    if value is None:
        return tuple()

    text = str(value).strip()

    if not text:
        return tuple()

    return tuple(
        label.strip()
        for label in text.split("|")
        if label.strip()
    )


def load_longitudinal_metadata(
    csv_path: str,
) -> pd.DataFrame:
    """
    Load original NIH metadata and normalize column names.
    """

    dataframe = pd.read_csv(
        csv_path
    )

    dataframe.columns = [
        column.strip()
        for column in dataframe.columns
    ]

    required = {
        "Image Index",
        "Finding Labels",
        "Patient ID",
        "Follow-up #",
        "View Position",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "NIH metadata missing required columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe.copy()

    dataframe[
        "Patient ID"
    ] = dataframe[
        "Patient ID"
    ].astype(str)

    dataframe[
        "Follow-up #"
    ] = pd.to_numeric(
        dataframe[
            "Follow-up #"
        ],
        errors="raise",
    ).astype(int)

    return dataframe
def dataframe_to_longitudinal_studies(
    dataframe: pd.DataFrame,
) -> list[NIHLongitudinalStudy]:

    studies: list[
        NIHLongitudinalStudy
    ] = []

    for _, row in dataframe.iterrows():

        view = row.get(
            "View Position"
        )

        if pd.isna(view):
            view_position = None
        else:
            view_position = str(
                view
            ).strip()

        study = NIHLongitudinalStudy(
            image_index=str(
                row[
                    "Image Index"
                ]
            ),

            patient_id=str(
                row[
                    "Patient ID"
                ]
            ),

            follow_up_number=int(
                row[
                    "Follow-up #"
                ]
            ),

            view_position=(
                view_position
            ),

            labels=normalize_labels(
                row[
                    "Finding Labels"
                ]
            ),
        )

        studies.append(
            study
        )

    return studies
def group_studies_by_patient(
    studies: list[
        NIHLongitudinalStudy
    ],
) -> dict[
    str,
    list[
        NIHLongitudinalStudy
    ],
]:
    groups: dict[
        str,
        list[
            NIHLongitudinalStudy
        ],
    ] = {}

    for study in studies:
        groups.setdefault(
            study.patient_id,
            [],
        ).append(
            study
        )

    for patient_id in groups:
        groups[
            patient_id
        ] = sorted(
            groups[
                patient_id
            ],
            key=lambda item: (
                item.follow_up_number,
                item.image_index,
            ),
        )

    return groups


def get_repeated_patient_groups(
    studies: list[
        NIHLongitudinalStudy
    ],
) -> dict[
    str,
    list[
        NIHLongitudinalStudy
    ],
]:

    groups = (
        group_studies_by_patient(
            studies
        )
    )

    return {
        patient_id: patient_studies
        for patient_id, patient_studies
        in groups.items()
        if len(
            patient_studies
        ) >= 2
    }
def build_adjacent_pairs(
    studies: list[
        NIHLongitudinalStudy
    ],
    same_view_only: bool = False,
) -> list[
    NIHLongitudinalPair
]:

    groups = (
        get_repeated_patient_groups(
            studies
        )
    )

    pairs: list[
        NIHLongitudinalPair
    ] = []

    for (
        patient_id,
        patient_studies,
    ) in groups.items():

        for index in range(
            len(patient_studies) - 1
        ):

            prior = (
                patient_studies[
                    index
                ]
            )

            current = (
                patient_studies[
                    index + 1
                ]
            )

            if (
                same_view_only
                and prior.view_position
                != current.view_position
            ):
                continue

            delta = (
                current.follow_up_number
                - prior.follow_up_number
            )

            if delta <= 0:
                continue

            pairs.append(
                NIHLongitudinalPair(
                    patient_id=(
                        patient_id
                    ),

                    prior_image_index=(
                        prior.image_index
                    ),

                    current_image_index=(
                        current.image_index
                    ),

                    prior_follow_up=(
                        prior.follow_up_number
                    ),

                    current_follow_up=(
                        current.follow_up_number
                    ),

                    prior_labels=(
                        prior.labels
                    ),

                    current_labels=(
                        current.labels
                    ),

                    prior_view=(
                        prior.view_position
                    ),

                    current_view=(
                        current.view_position
                    ),

                    follow_up_delta=(
                        delta
                    ),
                )
            )

    return pairs