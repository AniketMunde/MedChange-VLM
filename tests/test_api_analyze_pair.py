from io import BytesIO

from fastapi.testclient import (
    TestClient,
)
from PIL import Image

from medchange.api.app import (
    app,
)


client = TestClient(
    app
)


def _png_bytes(
    value: int,
) -> bytes:
    image = Image.new(
        "L",
        (64, 64),
        value,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_identical_images_rejected():
    content = _png_bytes(
        80
    )

    response = client.post(
        "/analyze-pair",

        files={
            "prior": (
                "prior.png",
                content,
                "image/png",
            ),

            "current": (
                "current.png",
                content,
                "image/png",
            ),
        },

        data={
            "pair_id":
                "api-test",

            "prior_study_id":
                "prior",

            "current_study_id":
                "current",

            "safety_policy":
                "change_sensitive",

            "safety_threshold":
                "0.80",
        },
    )

    assert (
        response.status_code
        == 400
    )

    detail = (
        response.json()[
            "detail"
        ]
    )

    assert (
        detail[
            "code"
        ]
        == "invalid_image_pair"
    )

    assert (
        "identical file content"
        in detail[
            "message"
        ].lower()
    )

    assert (
        detail[
            "retryable"
        ]
        is False
    )