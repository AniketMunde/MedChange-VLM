from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from medchange.inference.temporal_pipeline import (
    TemporalQwenPipeline,
)
from medchange.models.temporal.ablation_features import (
    build_ablation_vector,
)
from medchange.models.temporal.features import (
    build_temporal_embedding_features,
)
from medchange.models.vision import (
    BiomedCLIP,
)
from medchange.models.vlm.config import (
    VLMConfig,
)
from medchange.models.vlm.qwen_vl import (
    QwenVLM,
)
from medchange.reasoning.unified_pipeline import (
    build_unified_temporal_result,
)
from medchange.reporting.report_builder import (
    build_longitudinal_report,
)
from medchange.safety.validation import (
    validate_longitudinal_pair,
)
from medchange.safety.config import (
    SafetyPolicyConfig,
    VALID_SAFETY_POLICIES,
)


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete MedChange-VLM "
            "longitudinal inference pipeline."
        )
    )

    parser.add_argument(
        "--prior",
        required=True,
        help="Path to prior chest X-ray.",
    )

    parser.add_argument(
        "--current",
        required=True,
        help="Path to current chest X-ray.",
    )

    parser.add_argument(
        "--pair-id",
        required=True,
    )

    parser.add_argument(
        "--prior-study-id",
        required=True,
    )

    parser.add_argument(
        "--current-study-id",
        required=True,
    )

    parser.add_argument(
        "--classifier-dir",
        default="models/temporal",
    )

    parser.add_argument(
        "--output",
        default=(
            "experiments/m5/"
            "medchange_result.json"
        ),
    )
    parser.add_argument(
        "--safety-policy",
        choices=sorted(
            VALID_SAFETY_POLICIES
        ),
        default="change_sensitive",
        help=(
            "MedChange selective-risk "
            "operating policy."
        ),
    )

    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=None,
        help=(
            "Optional BiomedCLIP confidence "
            "threshold. If omitted, the "
            "policy-specific validated default "
            "is used."
        ),
    )

    return parser.parse_args()

def load_classifier_artifact(
    classifier_dir: Path,
    finding: str,
) -> dict:
    path = (
        classifier_dir
        / f"{finding}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Temporal classifier artifact "
            f"not found: {path}"
        )

    with path.open(
        "rb"
    ) as file:
        artifact = pickle.load(
            file
        )

    required = {
        "finding",
        "feature_set",
        "classifier",
    }

    missing = (
        required
        - set(
            artifact.keys()
        )
    )

    if missing:
        raise ValueError(
            f"Classifier artifact for "
            f"{finding} is missing fields: "
            f"{sorted(missing)}"
        )

    if (
        artifact[
            "finding"
        ]
        != finding
    ):
        raise ValueError(
            "Classifier artifact finding mismatch: "
            f"expected={finding}, "
            f"actual="
            f"{artifact['finding']}"
        )

    return artifact


def normalize_finding_name(
    name: str,
) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    )


def run_biomedclip_temporal(
    prior_path: Path,
    current_path: Path,
    classifier_dir: Path,
) -> dict:
    """
    Run frozen BiomedCLIP on both images and apply
    the pathology-specific temporal classifiers exported
    from M5.1.1.
    """

    print()
    print("=" * 90)
    print(
        "BIOMEDCLIP TEMPORAL INFERENCE"
    )
    print("=" * 90)

    print(
        "Loading BiomedCLIP..."
    )

    model = BiomedCLIP()

    print(
        "Encoding prior image..."
    )

    prior_embedding = (
        model.encode_image(
            prior_path
        )
    )

    print(
        "Encoding current image..."
    )

    current_embedding = (
        model.encode_image(
            current_path
        )
    )

    temporal_features = (
        build_temporal_embedding_features(
            prior_embedding,
            current_embedding,
        )
    )

    predictions = {}

    for finding in (
        TARGET_FINDINGS
    ):
        artifact = (
            load_classifier_artifact(
                classifier_dir=(
                    classifier_dir
                ),
                finding=finding,
            )
        )

        classifier = (
            artifact[
                "classifier"
            ]
        )

        feature_set = (
            artifact[
                "feature_set"
            ]
        )

        feature_vector = (
            build_ablation_vector(
                temporal_features,
                feature_set,
            )
        ).reshape(
            1,
            -1,
        )

        predicted_state = (
            classifier.predict(
                feature_vector
            )[0]
        )

        confidence = None

        # FusionClassifier wraps sklearn LogisticRegression
        # inside classifier.model and scaling inside
        # classifier.scaler.
        if (
            hasattr(
                classifier,
                "model",
            )
            and hasattr(
                classifier.model,
                "predict_proba",
            )
        ):
            scaled = (
                classifier.scaler
                .transform(
                    feature_vector
                )
            )

            probabilities = (
                classifier.model
                .predict_proba(
                    scaled
                )[0]
            )

            confidence = float(
                np.max(
                    probabilities
                )
            )

        predictions[
            finding
        ] = {
            "state": str(
                predicted_state
            ),

            "confidence": (
                confidence
            ),

            "feature_set": (
                feature_set
            ),
        }

        confidence_text = (
            f"{confidence:.3f}"
            if confidence
            is not None
            else "N/A"
        )

        print(
            f"{finding:<20} "
            f"{str(predicted_state):<12} "
            f"confidence="
            f"{confidence_text:<8} "
            f"features="
            f"{feature_set}"
        )

    return predictions

def print_longitudinal_report(
    report,
) -> None:
    print()
    print("=" * 110)
    print(
        "EVIDENCE-GROUNDED "
        "LONGITUDINAL REPORT"
    )
    print("=" * 110)

    print(
        f"Overall change : "
        f"{report.overall_change}"
    )

    print(
        f"Uncertainty    : "
        f"{report.overall_uncertainty}"
    )

    print(
        f"Review needed  : "
        f"{report.requires_review}"
    )

    print()
    print("IMPRESSION")
    print("-" * 110)
    print(
        report.impression
    )

    if report.new_findings:
        print()
        print("NEW FINDINGS")

        for item in (
            report.new_findings
        ):
            print(
                f"- {item.finding}"
            )

    if report.resolved_findings:
        print()
        print("RESOLVED FINDINGS")

        for item in (
            report.resolved_findings
        ):
            print(
                f"- {item.finding}"
            )

    if report.persistent_findings:
        print()
        print("PERSISTENT FINDINGS")

        for item in (
            report.persistent_findings
        ):
            print(
                f"- {item.finding}"
            )

    if report.uncertain_findings:
        print()
        print(
            "UNCERTAIN / CONFLICTING FINDINGS"
        )

        for item in (
            report.uncertain_findings
        ):
            print(
                f"- {item.finding}: "
                f"{item.evidence or 'No evidence text'}"
            )

    if report.review_notes:
        print()
        print("REVIEW FLAGS")

        for note in (
            report.review_notes
        ):
            print(
                f"- {note}"
            )

    print("=" * 110)


def extract_qwen_predictions(
    prediction,
) -> tuple[
    dict,
    dict,
]:
    """
    Convert TemporalPrediction into the dictionaries
    required by the MedChange reasoning layer.
    """

    qwen_predictions = {}
    explanations = {}

    for item in (
        prediction.findings
    ):
        finding = (
            normalize_finding_name(
                item.finding
            )
        )

        if finding not in (
            TARGET_FINDINGS
        ):
            continue

        qwen_predictions[
            finding
        ] = {
            "state": (
                item.change
            ),

            "confidence": float(
                item.confidence
            ),
        }

        explanations[
            finding
        ] = (
            item.evidence
        )

    return (
        qwen_predictions,
        explanations,
    )


def run_qwen_temporal(
    prior_path: Path,
    current_path: Path,
    pair_id: str,
    prior_study_id: str,
    current_study_id: str,
) -> tuple[
    dict,
    dict,
    dict,
]:
    print()
    print("=" * 90)
    print(
        "QWEN2.5-VL TEMPORAL REASONING"
    )
    print("=" * 90)

    config = VLMConfig()

    vlm = QwenVLM(
        config=config
    )

    pipeline = (
        TemporalQwenPipeline(
            vlm=vlm
        )
    )

    result = (
        pipeline
        .analyze_pair_detailed(
            prior_image_path=(
                prior_path
            ),

            current_image_path=(
                current_path
            ),

            pair_id=(
                pair_id
            ),

            prior_study_id=(
                prior_study_id
            ),

            current_study_id=(
                current_study_id
            ),

            save_raw_output=True,
        )
    )

    (
        qwen_predictions,
        explanations,
    ) = extract_qwen_predictions(
        result.prediction
    )

    print()

    for finding in (
        TARGET_FINDINGS
    ):
        payload = (
            qwen_predictions
            .get(
                finding
            )
        )

        if payload is None:
            print(
                f"{finding:<20} "
                "missing"
            )

            continue

        print(
            f"{finding:<20} "
            f"{payload['state']:<12} "
            f"confidence="
            f"{payload['confidence']:.3f}"
        )

    metrics = {
        "elapsed_seconds": (
            result.metrics
            .elapsed_seconds
        ),

        "gpu_peak_gb": (
            result.metrics
            .gpu_peak_allocated_gb
        ),

        "json_repaired": (
            result.json_repaired
        ),
    }

    return (
        qwen_predictions,
        explanations,
        metrics,
    )


def print_unified_result(
    result,
) -> None:
    print()
    print("=" * 110)
    print(
        "MEDCHANGE-VLM — "
        "UNIFIED TEMPORAL RESULT"
    )
    print("=" * 110)

    print(
        f"Pair            : "
        f"{result.pair_id}"
    )

    print(
        f"Prior study     : "
        f"{result.prior_study_id}"
    )

    print(
        f"Current study   : "
        f"{result.current_study_id}"
    )

    print(
        f"Overall change  : "
        f"{result.overall_change}"
    )

    print(
        f"Uncertainty     : "
        f"{result.uncertainty}"
    )

    print(
        f"Requires review : "
        f"{result.requires_review}"
    )

    print()

    print(
        f"{'Finding':<20}"
        f"{'Final':<13}"
        f"{'BiomedCLIP':<13}"
        f"{'Qwen':<13}"
        f"{'Agreement':<13}"
        f"{'Uncertainty':<13}"
        f"{'Review'}"
    )

    print("-" * 110)

    for finding in (
        result.findings
    ):
        print(
            f"{finding.finding:<20}"
            f"{finding.final_state:<13}"
            f"{str(finding.biomedclip_state):<13}"
            f"{str(finding.qwen_state):<13}"
            f"{finding.agreement:<13}"
            f"{finding.uncertainty:<13}"
            f"{finding.requires_review}"
        )

    print("-" * 110)

    print()
    print(
        "Summary:"
    )

    print(
        result.summary
    )

    print("=" * 110)


def main() -> None:
    args = parse_args()

    start = (
        time.perf_counter()
    )

    (
        prior_path,
        current_path,
    ) = validate_longitudinal_pair(
        args.prior,
        args.current,
    )

    classifier_dir = Path(
        args.classifier_dir
    )

    if not classifier_dir.exists():
        raise FileNotFoundError(
            "Temporal classifier directory "
            f"not found: "
            f"{classifier_dir}"
        )

    print()
    print("=" * 110)
    print(
        "MedChange-VLM"
    )

    print(
        "End-to-End Longitudinal "
        "Inference Engine"
    )
    print("=" * 110)

    print(
        f"Pair    : "
        f"{args.pair_id}"
    )

    print(
        f"Prior   : "
        f"{prior_path}"
    )

    print(
        f"Current : "
        f"{current_path}"
    )

    biomedclip_predictions = (
        run_biomedclip_temporal(
            prior_path=prior_path,
            current_path=current_path,
            classifier_dir=(
                classifier_dir
            ),
        )
    )

    (
        qwen_predictions,
        qwen_explanations,
        qwen_metrics,
    ) = run_qwen_temporal(
        prior_path=prior_path,
        current_path=current_path,

        pair_id=args.pair_id,

        prior_study_id=(
            args.prior_study_id
        ),

        current_study_id=(
            args.current_study_id
        ),
    )
    safety_config = (
        SafetyPolicyConfig(
            policy=(
                args.safety_policy
            ),
            threshold=(
                args.safety_threshold
            ),
        )
    )

    result = (
        build_unified_temporal_result(
            pair_id=args.pair_id,

            prior_study_id=(
                args.prior_study_id
            ),

            current_study_id=(
                args.current_study_id
            ),

            biomedclip_predictions=(
                biomedclip_predictions
            ),

            qwen_predictions=(
                qwen_predictions
            ),

            qwen_explanations=(
                qwen_explanations
            ),

            safety_config=(
                safety_config
            ),
        )
    )
    report = (
        build_longitudinal_report(
            result
        )
    )

    total_elapsed = (
        time.perf_counter()
        - start
    )


    print_unified_result(
        result
    )
    print_longitudinal_report(
        report
    )

    output_payload = (
        result.model_dump()
    )
    output_payload[
        "longitudinal_report"
    ] = report.model_dump()

    output_payload[
        "model_evidence"
    ] = {
        "biomedclip": (
            biomedclip_predictions
        ),

        "qwen": (
            qwen_predictions
        ),
    }

    output_payload[
        "runtime"
    ] = {
        "total_elapsed_seconds": (
            total_elapsed
        ),

        "qwen": (
            qwen_metrics
        ),
    }
    output_payload[
        "safety_policy"
    ] = {
        "policy": (
            safety_config.policy
        ),
        "threshold": (
            safety_config.threshold
        ),
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output_payload,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print(
        f"Safety policy : "
        f"{safety_config.policy}"
    )

    print(
        f"Threshold     : "
        f"{safety_config.threshold:.2f}"
    )
    print(
        f"Total elapsed : "
        f"{total_elapsed:.2f}s"
    )

    print(
        f"Output        : "
        f"{output_path}"
    )



if __name__ == "__main__":
    main()