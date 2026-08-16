from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from medchange.data.nih.longitudinal import (
    dataframe_to_longitudinal_studies,
    load_longitudinal_metadata,
)
from medchange.data.nih.longitudinal_audit import (
    audit_longitudinal_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit NIH ChestXray14 "
            "for longitudinal pairing."
        )
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=(
            "data/nih/"
            "Data_Entry_2017.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "experiments/"
            "nih_longitudinal/"
            "audit.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = (
        load_longitudinal_metadata(
            args.metadata
        )
    )

    studies = (
        dataframe_to_longitudinal_studies(
            dataframe
        )
    )

    audit = (
        audit_longitudinal_dataset(
            studies
        )
    )

    print()
    print("=" * 80)
    print(
        "NIH ChestXray14 "
        "Longitudinal Feasibility Audit"
    )
    print("=" * 80)

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved audit to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()