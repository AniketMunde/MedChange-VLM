import time

from medchange.inference.metrics import (
    InferenceTimer,
)


def test_inference_timer():
    timer = InferenceTimer()

    timer.start()

    time.sleep(0.01)

    metrics = timer.stop()

    assert metrics.elapsed_seconds > 0