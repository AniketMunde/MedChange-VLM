from medchange.models.vision.prompts import (
    build_prompt_pair,
)
from medchange.models.vision.prompts import (
    display_finding,
)


def test_display_finding():
    assert (
        display_finding(
            "pleural_effusion"
        )
        == "pleural effusion"
    )


def test_prompt_pair():
    positive, negative = (
        build_prompt_pair(
            "pleural effusion"
        )
    )

    assert (
        "pleural effusion"
        in positive
    )

    assert (
        "pleural effusion"
        in negative
    )

    assert "showing" in positive

    assert "without" in negative