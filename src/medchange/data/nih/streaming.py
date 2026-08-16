from __future__ import annotations

from collections.abc import Iterator

from datasets import (
    IterableDataset,
    get_dataset_split_names,
    load_dataset,
)

from medchange.data.nih.adapter import (
    NIHExample,
    adapt_nih_example,
)
from medchange.data.nih.constants import (
    HF_NIH_DATASET,
    SUPPORTED_VIEWS,
)


def get_nih_split_names() -> list[str]:
    """
    Query Hugging Face for the actual split names.

    We deliberately avoid hard-coding split assumptions.
    """

    return list(
        get_dataset_split_names(
            HF_NIH_DATASET
        )
    )


def load_nih_stream(
    split: str,
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 1000,
) -> IterableDataset:
    """
    Load NIH ChestXray14 as a streaming IterableDataset.
    """

    available_splits = (
        get_nih_split_names()
    )

    if split not in available_splits:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Available splits: {available_splits}"
        )

    dataset = load_dataset(
        HF_NIH_DATASET,
        split=split,
        streaming=True,
    )

    if shuffle:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer,
        )

    return dataset


def iter_nih_examples(
    split: str,
    max_samples: int | None = None,
    view: str = "ALL",
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 1000,
) -> Iterator[NIHExample]:
    """
    Stream adapted NIH examples.

    Parameters
    ----------
    view:
        ALL, AP or PA.

    max_samples:
        Maximum number of accepted examples, not raw rows.
    """

    normalized_view = (
        view.strip().upper()
    )

    allowed = (
        SUPPORTED_VIEWS
        | {"ALL"}
    )

    if normalized_view not in allowed:
        raise ValueError(
            "view must be one of "
            f"{sorted(allowed)}"
        )

    if (
        max_samples is not None
        and max_samples <= 0
    ):
        raise ValueError(
            "max_samples must be positive."
        )

    dataset = load_nih_stream(
        split=split,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )

    accepted = 0

    for raw_sample in dataset:
        example = adapt_nih_example(
            raw_sample
        )

        if (
            normalized_view != "ALL"
            and example.view_position
            != normalized_view
        ):
            continue

        yield example

        accepted += 1

        if (
            max_samples is not None
            and accepted >= max_samples
        ):
            break