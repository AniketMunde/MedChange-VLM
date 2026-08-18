import torch

from medchange.training.temporal_vlm_collator import (
    TemporalVLMCollator,
)


def test_assistant_only_labels():
    input_ids = torch.tensor(
        [
            [
                10,
                11,
                12,
                20,
                21,
                22,
                0,
                0,
            ]
        ]
    )

    attention_mask = torch.tensor(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
            ]
        ]
    )

    labels = (
        TemporalVLMCollator
        ._build_assistant_only_labels(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
            ),
            prompt_lengths=[
                3
            ],
            pad_token_id=0,
        )
    )

    expected = torch.tensor(
        [
            [
                -100,
                -100,
                -100,
                20,
                21,
                22,
                -100,
                -100,
            ]
        ]
    )

    assert torch.equal(
        labels,
        expected,
    )


def test_different_prompt_lengths():
    input_ids = torch.tensor(
        [
            [
                1,
                2,
                3,
                4,
                5,
            ],

            [
                6,
                7,
                8,
                9,
                10,
            ],
        ]
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    labels = (
        TemporalVLMCollator
        ._build_assistant_only_labels(
            input_ids=input_ids,
            attention_mask=(
                attention_mask
            ),
            prompt_lengths=[
                2,
                4,
            ],
            pad_token_id=None,
        )
    )

    assert labels[
        0
    ].tolist() == [
        -100,
        -100,
        3,
        4,
        5,
    ]

    assert labels[
        1
    ].tolist() == [
        -100,
        -100,
        -100,
        -100,
        10,
    ]