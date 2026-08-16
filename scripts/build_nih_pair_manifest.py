from __future__ import annotations

import argparse
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
    build_adjacent_pairs,
    dataframe_to_longitudinal_studies,
    load_longitudinal_metadata,
)
from medchange.data.nih.pair_manifest import (
    build_pair_manifest,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        default=(
            "data/nih/"
            "Data_Entry_2017.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/nih/"
            "longitudinal_pairs.csv"
        ),
    )

    parser.add_argument(
        "--same-view-only",
        action="store_true",
    )

    args = parser.parse_args()

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

    pairs = (
        build_adjacent_pairs(
            studies,
            same_view_only=(
                args.same_view_only
            ),
        )
    )

    manifest = (
        build_pair_manifest(
            pairs
        )
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MedChange-VLM "
        "Longitudinal Pair Manifest"
    )

    print(
        "=" * 70
    )

    print(
        f"Pairs       : "
        f"{len(manifest):,}"
    )

    print(
        f"Patients    : "
        f"{manifest['patient_id'].nunique():,}"
    )

    print(
        f"Same view   : "
        f"{manifest['same_view'].sum():,}"
    )

    print(
        f"Label change: "
        f"{manifest['has_label_change'].sum():,}"
    )

    print(
        f"Output      : "
        f"{output}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()