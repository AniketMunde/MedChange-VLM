from fastapi.testclient import (
    TestClient,
)

from medchange.api.app import (
    app,
)


client = TestClient(
    app
)


def test_model_info():
    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "safety_policy"
        ]
        == "change_sensitive"
    )

    assert (
        payload[
            "safety_threshold"
        ]
        == 0.80
    )

    assert (
        len(
            payload[
                "target_findings"
            ]
        )
        == 7
    )