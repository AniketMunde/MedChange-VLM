from medchange.models.vlm.temporal_prompts import (
    build_temporal_comparison_prompt,
)


def test_temporal_prompt():

    prompt = (
        build_temporal_comparison_prompt()
    )

    assert (
        "PRIOR"
        in prompt
    )

    assert (
        "CURRENT"
        in prompt
    )

    assert (
        "new"
        in prompt
    )

    assert (
        "resolved"
        in prompt
    )

    assert (
        "pneumothorax"
        in prompt
    )