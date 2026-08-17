from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from medchange.api.app import app


client = TestClient(
    app
)


def _png_bytes(
    value: int = 80,
) -> bytes:
    image = Image.new(
        "L",
        (128, 128),
        value,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def _valid_files():
    return {
        "prior": (
            "prior.png",
            _png_bytes(
                50
            ),
            "image/png",
        ),

        "current": (
            "current.png",
            _png_bytes(
                100
            ),
            "image/png",
        ),
    }


def _valid_data():
    return {
        "pair_id": "test-pair",
        "prior_study_id": "prior-1",
        "current_study_id": "current-1",
        "safety_policy": (
            "change_sensitive"
        ),
        "safety_threshold": "0.80",
    }


def test_missing_pair_id():
    data = _valid_data()

    del data[
        "pair_id"
    ]

    response = client.post(
        "/analyze-pair",
        files=_valid_files(),
        data=data,
    )

    assert (
        response.status_code
        == 422
    )


def test_empty_pair_id():
    data = _valid_data()

    data[
        "pair_id"
    ] = "   "

    response = client.post(
        "/analyze-pair",
        files=_valid_files(),
        data=data,
    )

    assert (
        response.status_code
        == 422
    )

    detail = response.json()[
        "detail"
    ]

    assert (
        detail[
            "code"
        ]
        == "invalid_pair_id"
    )


def test_identical_study_ids():
    data = _valid_data()

    data[
        "current_study_id"
    ] = (
        data[
            "prior_study_id"
        ]
    )

    response = client.post(
        "/analyze-pair",
        files=_valid_files(),
        data=data,
    )

    assert (
        response.status_code
        == 422
    )


def test_unsupported_content_type():
    files = {
        "prior": (
            "prior.txt",
            b"hello",
            "text/plain",
        ),

        "current": (
            "current.png",
            _png_bytes(),
            "image/png",
        ),
    }

    response = client.post(
        "/analyze-pair",
        files=files,
        data=_valid_data(),
    )

    assert (
        response.status_code
        == 415
    )

    assert (
        response.json()[
            "detail"
        ][
            "code"
        ]
        == "unsupported_prior_type"
    )


def test_invalid_safety_policy():
    data = _valid_data()

    data[
        "safety_policy"
    ] = "bad_policy"

    response = client.post(
        "/analyze-pair",
        files=_valid_files(),
        data=data,
    )

    assert (
        response.status_code
        == 422
    )

    assert (
        response.json()[
            "detail"
        ][
            "code"
        ]
        == "invalid_safety_policy"
    )