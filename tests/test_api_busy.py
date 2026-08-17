from __future__ import annotations

import importlib
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from medchange.runtime import (
    RuntimeBusyError,
)


app_module = importlib.import_module(
    "medchange.api.app"
)

app = app_module.app

client = TestClient(
    app
)


def _png_bytes(
    value: int,
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


def test_busy_runtime_returns_503(
    monkeypatch,
):
    files = {
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

    data = {
        "pair_id": (
            "busy-test"
        ),

        "prior_study_id": (
            "p"
        ),

        "current_study_id": (
            "c"
        ),

        "safety_policy": (
            "change_sensitive"
        ),

        "safety_threshold": (
            "0.80"
        ),
    }

    class BusyRuntime:
        def record_cache_hit(
            self,
        ) -> None:
            pass

        def run(
            self,
            *args,
            **kwargs,
        ):
            raise RuntimeBusyError(
                "busy"
            )

    monkeypatch.setattr(
        app_module,
        "get_runtime_manager",
        lambda: BusyRuntime(),
    )

    response = client.post(
        "/analyze-pair",
        files=files,
        data=data,
    )

    assert (
        response.status_code
        == 503
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
        == "inference_busy"
    )

    assert (
        detail[
            "retryable"
        ]
        is True
    )

    assert (
        detail[
            "message"
        ]
        == (
            "MedChange inference engine "
            "is currently busy."
        )
    )