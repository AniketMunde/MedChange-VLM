from __future__ import annotations

from datetime import datetime
from pathlib import Path


def save_raw_vlm_output(
    output: str,
    study_id: str,
    output_dir: str | Path = "logs/vlm_outputs",
) -> Path:
    """
    Persist raw VLM output for debugging and reproducibility.
    """

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_study_id = study_id.replace("/", "_").replace("\\", "_")

    path = directory / f"{safe_study_id}_{timestamp}.txt"

    path.write_text(
        output,
        encoding="utf-8",
    )

    return path