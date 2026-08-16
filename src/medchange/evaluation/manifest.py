from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_BASE_COLUMNS = {
    "image_path",
    "study_id",
}


def load_evaluation_manifest(
    manifest_path: str | Path,
) -> pd.DataFrame:
    """
    Load an evaluation CSV manifest.
    """

    path = Path(
        manifest_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    missing = (
        REQUIRED_BASE_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Manifest missing required columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Evaluation manifest is empty."
        )

    return dataframe


def validate_image_paths(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure all paths in an evaluation manifest exist.
    """

    missing_paths = []

    for path_string in dataframe[
        "image_path"
    ]:
        path = Path(
            str(path_string)
        )

        if not path.exists():
            missing_paths.append(
                str(path)
            )

    if missing_paths:
        preview = missing_paths[
            :5
        ]

        raise FileNotFoundError(
            "Manifest contains missing images. "
            f"Examples: {preview}"
        )