from __future__ import annotations


TARGET_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural effusion",
    "pneumonia",
    "pneumothorax",
]


def build_temporal_comparison_prompt() -> str:
    findings = "\n".join(
        f"- {finding}"
        for finding
        in TARGET_FINDINGS
    )

    return f"""
You are analyzing TWO chest radiographs from the SAME patient.

Image 1 is the PRIOR chest radiograph.
Image 2 is the CURRENT chest radiograph.

Compare the two images directly.

For each of the following findings:

{findings}

classify the temporal state as exactly one of:

- new
- resolved
- persistent
- absent
- uncertain

Definitions:

new:
The finding is absent on the prior image and present on the current image.

resolved:
The finding is present on the prior image and absent on the current image.

persistent:
The finding is present on both images.

absent:
The finding is absent on both images.

uncertain:
The images do not allow a confident temporal assessment.

Also provide:

- prior_status
- current_status
- confidence from 0.0 to 1.0
- evidence: maximum 12 words, visual comparison only
- overall_change
- Keep every evidence field under 12 words.
- Keep the summary under 25 words.

Important rules:

1. Compare the PRIOR and CURRENT images directly.
2. Do not infer exact elapsed time.
3. Do not invent findings that are not visually supported.
4. If projection or image quality makes comparison unreliable, use uncertain.
5. Do not provide clinical management advice.
6. Return ONLY valid JSON.
7. Use exactly the requested schema.

Return:

{{
  "findings": [
    {{
      "finding": "atelectasis",
      "prior_status": "absent",
      "current_status": "present",
      "change": "new",
      "confidence": 0.82,
      "evidence": "..."
    }}
  ],
  "overall_change": "mixed",
  "summary": "..."
}}
""".strip()