from medchange.evaluation.calibration import (
    find_best_f1_threshold,
)


def test_threshold_calibration():
    result = (
        find_best_f1_threshold(
            y_true=[
                0,
                0,
                1,
                1,
            ],

            y_score=[
                0.10,
                0.20,
                0.35,
                0.40,
            ],
        )
    )

    assert (
        0.20
        < result.threshold
        <= 0.40
    )

    assert result.f1 > 0.0