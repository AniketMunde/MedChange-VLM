from medchange.runtime.manager import (
    InferenceRuntimeManager,
)


def test_runtime_success():
    manager = (
        InferenceRuntimeManager()
    )

    result = manager.run(
        "test",
        lambda: 42,
    )

    assert result == 42

    status = manager.status()

    assert (
        status.total_requests
        == 1
    )

    assert (
        status.successful_requests
        == 1
    )

    assert (
        status.failed_requests
        == 0
    )

    assert (
        status.busy
        is False
    )


def test_runtime_failure():
    manager = (
        InferenceRuntimeManager()
    )

    def fail():
        raise ValueError(
            "test"
        )

    try:
        manager.run(
            "test",
            fail,
        )

    except ValueError:
        pass

    status = (
        manager.status()
    )

    assert (
        status.failed_requests
        == 1
    )