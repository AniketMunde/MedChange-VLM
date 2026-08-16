from medchange.models.vlm.prompts import (
    build_single_image_prompt,
)


def test_default_prompt():
    prompt = (
        build_single_image_prompt()
    )

    assert "JSON" in prompt

    assert "findings" in prompt

    assert "confidence" in prompt


def test_question_added():
    prompt = (
        build_single_image_prompt(
            "Is pleural effusion visible?"
        )
    )

    assert (
        "Is pleural effusion visible?"
        in prompt
    )