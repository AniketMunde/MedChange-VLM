from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final MedChange-VLM API smoke test."
        )
    )

    parser.add_argument(
        "--prior",
        required=True,
    )

    parser.add_argument(
        "--current",
        required=True,
    )

    parser.add_argument(
        "--api-url",
        default=(
            "http://127.0.0.1:8000"
        ),
    )

    parser.add_argument(
        "--pair-id",
        default="smoke-test",
    )

    parser.add_argument(
        "--prior-study-id",
        default="prior",
    )

    parser.add_argument(
        "--current-study-id",
        default="current",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prior_path = Path(
        args.prior
    )

    current_path = Path(
        args.current
    )

    if not prior_path.exists():
        raise FileNotFoundError(
            prior_path
        )

    if not current_path.exists():
        raise FileNotFoundError(
            current_path
        )

    health = requests.get(
        f"{args.api_url}/health",
        timeout=10,
    )

    health.raise_for_status()

    print(
        "Health:",
        health.json(),
    )

    with (
        prior_path.open(
            "rb"
        ) as prior_file,

        current_path.open(
            "rb"
        ) as current_file,
    ):
        response = requests.post(
            (
                f"{args.api_url}/"
                "analyze-pair"
            ),

            files={
                "prior": (
                    prior_path.name,
                    prior_file,
                    "image/png",
                ),

                "current": (
                    current_path.name,
                    current_file,
                    "image/png",
                ),
            },

            data={
                "pair_id": (
                    args.pair_id
                ),

                "prior_study_id": (
                    args.prior_study_id
                ),

                "current_study_id": (
                    args.current_study_id
                ),

                "safety_policy": (
                    "change_sensitive"
                ),

                "safety_threshold": (
                    "0.80"
                ),
            },

            timeout=240,
        )

    print(
        "Status:",
        response.status_code,
    )

    if (
        response.status_code
        != 200
    ):
        print(
            response.text
        )

        raise SystemExit(
            1
        )

    payload = (
        response.json()
    )

    required = {
        "pair_id",
        "overall_change",
        "uncertainty",
        "requires_review",
        "findings",
        "impression",
        "safety_policy",
        "safety_threshold",
        "cache_hit",
    }

    missing = (
        required
        - set(
            payload
        )
    )

    if missing:
        raise RuntimeError(
            "API response missing fields: "
            f"{sorted(missing)}"
        )

    if len(
        payload[
            "findings"
        ]
    ) != 7:
        raise RuntimeError(
            "Expected 7 target findings, "
            f"received "
            f"{len(payload['findings'])}."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "MEDCHANGE-VLM FINAL SMOKE TEST"
    )

    print(
        "=" * 80
    )

    print(
        "Pair:",
        payload[
            "pair_id"
        ],
    )

    print(
        "Overall change:",
        payload[
            "overall_change"
        ],
    )

    print(
        "Uncertainty:",
        payload[
            "uncertainty"
        ],
    )

    print(
        "Review:",
        payload[
            "requires_review"
        ],
    )

    print(
        "Safety policy:",
        payload[
            "safety_policy"
        ],
    )

    print(
        "Threshold:",
        payload[
            "safety_threshold"
        ],
    )

    print(
        "Cache hit:",
        payload[
            "cache_hit"
        ],
    )

    print()
    print(
        "Findings:"
    )

    for finding in (
        payload[
            "findings"
        ]
    ):
        print(
            f"  "
            f"{finding['finding']:<20} "
            f"{finding['final_state']:<12} "
            f"review="
            f"{finding['requires_review']}"
        )

    print()
    print(
        "Smoke test PASSED"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()