from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from medchange.data.nih import (
    TARGET_FINDINGS,
)
from medchange.data.nih.subset import (
    patient_aware_split,
)
from medchange.evaluation.aggregate import (
    compute_macro_metrics,
)
from medchange.evaluation.benchmark import (
    add_confidence_intervals,
    calibrate_thresholds,
    evaluate_test_with_thresholds,
)
from medchange.evaluation.dataset_summary import (
    build_dataset_summary,
)
from medchange.evaluation.nih_runner import (
    NIHEvaluationRunner,
)
from medchange.evaluation.stratified import (
    evaluate_by_view,
)
from medchange.evaluation.audit import (
    audit_benchmark,
    benchmark_audit_to_dict,
)

from medchange.evaluation.audit_report import (
    print_benchmark_audit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible patient-aware "
            "NIH BiomedCLIP benchmark."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--development-fraction",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/"
            "nih_biomedclip_m33"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 90)
    print(
        "MedChange-VLM — M3.3 "
        "Reproducible Biomedical Benchmark"
    )
    print("=" * 90)

    runner = (
        NIHEvaluationRunner()
    )

    predictions = runner.run(
        split=args.split,

        max_samples=(
            args.samples
        ),

        view="ALL",

        # Important:
        # sequential streaming avoids the network
        # instability encountered earlier.
        shuffle=False,

        seed=args.seed,
    )

    split = patient_aware_split(
        dataframe=predictions,
        development_fraction=(
            args.development_fraction
        ),
        seed=args.seed,
    )

    development = (
        split.development
    )

    test = (
        split.test
    )

    thresholds = (
        calibrate_thresholds(
            development=development,
            findings=TARGET_FINDINGS,
        )
    )

    test_metrics = (
        evaluate_test_with_thresholds(
            test=test,
            threshold_table=thresholds,
            findings=TARGET_FINDINGS,
        )
    )
    audit = audit_benchmark(
        development=development,
        test=test,
        threshold_table=thresholds,
        findings=TARGET_FINDINGS,
        requested_samples=args.samples,
        seed=args.seed,
        development_fraction=(
            args.development_fraction
        ),
    )

    print_benchmark_audit(
        audit
    )

    test_metrics = (
        add_confidence_intervals(
            metrics=test_metrics,
            test=test,
            findings=TARGET_FINDINGS,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
    )

    threshold_map = {
        row["finding"]: float(
            row["threshold"]
        )
        for _, row
        in thresholds.iterrows()
    }

    view_metrics = (
        evaluate_by_view(
            dataframe=test,
            labels=TARGET_FINDINGS,
            thresholds=threshold_map,
        )
    )

    macro_metrics = (
        compute_macro_metrics(
            test_metrics
        )
    )

    development_summary = (
        build_dataset_summary(
            development,
            TARGET_FINDINGS,
        )
    )

    test_summary = (
        build_dataset_summary(
            test,
            TARGET_FINDINGS,
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_dir
        / "all_predictions.csv",
        index=False,
    )

    development.to_csv(
        output_dir
        / "development_predictions.csv",
        index=False,
    )

    test.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
    )

    thresholds.to_csv(
        output_dir
        / "calibrated_thresholds.csv",
        index=False,
    )

    test_metrics.to_csv(
        output_dir
        / "test_metrics.csv",
        index=False,
    )

    view_metrics.to_csv(
        output_dir
        / "view_metrics.csv",
        index=False,
    )

    (
        output_dir
        / "macro_metrics.json"
    ).write_text(
        json.dumps(
            macro_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "development_summary.json"
    ).write_text(
        json.dumps(
            development_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "test_summary.json"
    ).write_text(
        json.dumps(
            test_summary,
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_path = (
            output_dir
            / "benchmark_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            benchmark_audit_to_dict(
                audit
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print(
        "FINAL TEST METRICS"
    )
    print("=" * 110)

    print(
        test_metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "MACRO METRICS"
    )

    print(
        json.dumps(
            macro_metrics,
            indent=2,
        )
    )

    print()
    print(
        f"Development images : "
        f"{len(development)}"
    )

    print(
        f"Test images        : "
        f"{len(test)}"
    )

    print(
        f"Artifacts          : "
        f"{output_dir}"
    )

    print("=" * 110)

    print(
        f"Audit              : "
        f"{audit_path}"
    )


if __name__ == "__main__":
    main()