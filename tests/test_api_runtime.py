from fastapi.testclient import (
    TestClient,
)

from medchange.api.app import (
    app,
)


client = TestClient(
    app
)


def test_runtime_status():
    response = client.get(
        "/runtime-status"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        "busy"
        in payload
    )

    assert (
        "cache_entries"
        in payload
    )


def test_cache_clear():
    response = client.post(
        "/cache/clear"
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "status"
        ]
        == "cleared"
    )