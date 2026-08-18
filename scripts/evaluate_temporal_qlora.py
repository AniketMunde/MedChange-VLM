from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from peft import PeftModel

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


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


from medchange.evaluation.qlora_temporal import (
    compute_temporal_metrics,
    exact_pair_match_rate,
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-jsonl",
        default=(
            "data/nih/"
            "qlora_smoke/"
            "test.jsonl"
        ),
    )

    parser.add_argument(
        "--adapter",
        default=None,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def load_jsonl(
    path: str | Path,
):
    records = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(
                    json.loads(
                        line
                    )
                )

    return records


def extract_json(
    text: str,
) -> dict:
    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start < 0
        or end < start
    ):
        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        text[
            start:
            end + 1
        ]
    )


def normalize_prediction(
    payload: dict,
) -> dict[str, str]:
    output = {}

    findings = payload.get(
        "findings",
        [],
    )

    for item in findings:
        name = (
            str(
                item.get(
                    "finding",
                    "",
                )
            )
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        state = (
            str(
                item.get(
                    "change",
                    "absent",
                )
            )
            .strip()
            .lower()
        )

        if (
            name
            in TARGET_FINDINGS
        ):
            output[
                name
            ] = state

    return output


def run_example(
    *,
    model,
    processor,
    record,
):
    from PIL import Image

    images = [
        Image.open(
            path
        ).convert(
            "RGB"
        )

        for path
        in record[
            "images"
        ]
    ]

    user_messages = (
        record[
            "messages"
        ][
            :2
        ]
    )

    prompt = (
        processor
        .apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    inputs = processor(
        text=[
            prompt
        ],
        images=[
            images
        ],
        return_tensors="pt",
    )

    inputs = {
        key: value.to(
            model.device
        )
        if hasattr(
            value,
            "to"
        )
        else value

        for key, value
        in inputs.items()
    }

    with torch.no_grad():
        generated = (
            model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
            )
        )

    generated_only = generated[
        :,
        inputs[
            "input_ids"
        ].shape[1]:
    ]

    text = (
        processor
        .batch_decode(
            generated_only,
            skip_special_tokens=True,
        )[0]
    )

    return text


def main():
    args = parse_args()

    model_name = (
        "Qwen/"
        "Qwen2.5-VL-3B-Instruct"
    )

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.float16
            ),
        )
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            model_name,
            min_pixels=224 * 224,
            max_pixels=224 * 224,
        )
    )

    print(
        "Loading base Qwen..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            model_name,
            quantization_config=(
                quantization
            ),
            device_map="auto",
            torch_dtype=(
                torch.float16
            ),
        )
    )

    if args.adapter:
        print(
            f"Loading adapter: "
            f"{args.adapter}"
        )

        model = (
            PeftModel
            .from_pretrained(
                model,
                args.adapter,
            )
        )

    model.eval()

    records = load_jsonl(
        args.test_jsonl
    )

    rows = []

    y_true = []
    y_pred = []
    pair_ids = []

    successful = 0

    start_time = (
        time.perf_counter()
    )

    for index, record in enumerate(
        records,
        start=1,
    ):
        raw_output = ""

        parse_success = False

        try:
            raw_output = (
                run_example(
                    model=model,
                    processor=processor,
                    record=record,
                )
            )

            payload = (
                extract_json(
                    raw_output
                )
            )

            prediction = (
                normalize_prediction(
                    payload
                )
            )

            parse_success = True
            successful += 1

        except Exception as exc:
            prediction = {}

            print(
                f"[{index}] failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        target_lookup = {
            item[
                "finding"
            ]:
            item[
                "change"
            ]

            for item
            in record[
                "target"
            ][
                "findings"
            ]
        }

        for finding in (
            TARGET_FINDINGS
        ):
            true_state = (
                target_lookup[
                    finding
                ]
            )

            predicted_state = (
                prediction.get(
                    finding,
                    "invalid",
                )
            )

            pair_ids.append(
                str(
                    record[
                        "pair_id"
                    ]
                )
            )

            y_true.append(
                true_state
            )

            y_pred.append(
                predicted_state
            )

        rows.append(
            {
                "pair_id":
                    record[
                        "pair_id"
                    ],

                "parse_success":
                    parse_success,

                "ground_truth":
                    target_lookup,

                "prediction":
                    prediction,

                "raw_output":
                    raw_output,
            }
        )

        print(
            f"[{index}/{len(records)}] "
            f"parse={parse_success}"
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = (
        compute_temporal_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    metrics[
        "exact_pair_match_rate"
    ] = (
        exact_pair_match_rate(
            pair_ids=pair_ids,
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    metrics[
        "parse_success_rate"
    ] = (
        successful
        / len(
            records
        )
    )

    metrics[
        "num_pairs"
    ] = len(
        records
    )

    metrics[
        "elapsed_seconds"
    ] = elapsed

    metrics[
        "adapter"
    ] = (
        args.adapter
        or "none"
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            {
                "metrics":
                    metrics,

                "predictions":
                    rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)

    print(
        "TEMPORAL QWEN EVALUATION"
    )

    print("=" * 90)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print("=" * 90)


if __name__ == "__main__":
    main()