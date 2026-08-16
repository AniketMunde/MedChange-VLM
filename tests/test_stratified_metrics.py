import pandas as pd

from medchange.evaluation.stratified import (
    evaluate_by_view,
)


def test_evaluate_by_view():
    dataframe = pd.DataFrame(
        {
            "view_position": [
                "PA",
                "PA",
                "AP",
                "AP",
            ],

            "pleural_effusion_label": [
                0,
                1,
                0,
                1,
            ],

            "pleural_effusion_score": [
                0.1,
                0.9,
                0.2,
                0.8,
            ],
        }
    )

    result = evaluate_by_view(
        dataframe=dataframe,

        labels=[
            "pleural_effusion"
        ],

        thresholds={
            "pleural_effusion": 0.5
        },
    )

    assert len(
        result
    ) == 2

    assert set(
        result[
            "view_position"
        ]
    ) == {
        "AP",
        "PA",
    }