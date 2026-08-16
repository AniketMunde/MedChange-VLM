from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from medchange.data.nih.image_resolver import (
    NIHImageResolver,
)
from medchange.models.temporal.embedding_cache import (
    EmbeddingCache,
)
from medchange.models.vision import (
    BiomedCLIP,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache missing BiomedCLIP embeddings "
            "for temporal fusion pairs."
        )
    )

    parser.add_argument(
        "--pairs",
        default=(
            "data/nih/"
            "fusion_qwen_subset_200.csv"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        default=(
            "data/nih/"
            "embedding_cache"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataframe = pd.read_csv(
        args.pairs
    )

    images = sorted(
        set(
            dataframe[
                "prior_image_index"
            ].astype(str)
        )
        |
        set(
            dataframe[
                "current_image_index"
            ].astype(str)
        )
    )

    resolver = NIHImageResolver(
        args.dataset_root
    )

    resolver.build_index()

    cache = EmbeddingCache(
        args.cache_dir
    )

    missing = [
        image_index
        for image_index in images
        if not cache.contains(
            image_index
        )
    ]

    print()
    print("=" * 80)
    print(
        "M4.5.2 — BIOMEDCLIP EMBEDDING CACHE"
    )
    print("=" * 80)

    print(
        f"Unique images     : "
        f"{len(images)}"
    )

    print(
        f"Already cached    : "
        f"{len(images) - len(missing)}"
    )

    print(
        f"Missing embeddings: "
        f"{len(missing)}"
    )

    if not missing:
        print(
            "Nothing to cache."
        )

        return

    model = BiomedCLIP()

    total = len(
        missing
    )

    for position, image_index in enumerate(
        missing,
        start=1,
    ):

        image_path = resolver.resolve(
            image_index
        )

        embedding = model.encode_image(
            image_path
        )

        cache.save(
            image_index,
            embedding,
        )

        if (
            position == 1
            or position % 25 == 0
            or position == total
        ):
            print(
                f"[{position}/{total}] "
                f"cached {image_index}"
            )

    remaining = [
        image_index
        for image_index in images
        if not cache.contains(
            image_index
        )
    ]

    print()
    print(
        f"Remaining missing : "
        f"{len(remaining)}"
    )

    if remaining:
        raise RuntimeError(
            "Some BiomedCLIP embeddings "
            "were not cached successfully."
        )

    print(
        "BiomedCLIP fusion cache complete."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()