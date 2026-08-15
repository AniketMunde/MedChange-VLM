from medchange.evaluation.agreement import (
    compare_qwen_biomedclip,
)
from medchange.schemas import (
    MedicalFinding,
    MedChangePrediction,
)


def test_model_agreement():
    prediction = MedChangePrediction(
        study_id="test-study",
        findings=[
            MedicalFinding(
                name="pleural effusion",
                present=True,
                confidence=0.8,
            )
        ],
    )

    scores = {
        "pleural_effusion": 0.75,
        "pneumothorax": 0.10,
    }

    result = (
        compare_qwen_biomedclip(
            prediction,
            scores,
        )
    )

    assert len(result) == 2

    assert (
        result[0].finding
        == "pleural_effusion"
    )

    assert (
        result[0].agreement
        == "agree"
    )