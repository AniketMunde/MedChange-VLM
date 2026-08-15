from __future__ import annotations


SYSTEM_PROMPT = """
You are assisting with a biomedical AI research experiment involving
chest radiographs.

Analyze only visible image evidence.

Do not invent clinical history, laboratory values, diagnoses, symptoms,
patient demographics, or treatments that are not directly available.

If a finding is uncertain, state that it is uncertain.

This system is for research and model evaluation only and must not be
treated as a clinical diagnosis.
""".strip()


CHEST_XRAY_ANALYSIS_PROMPT = """
Analyze this chest radiograph.

Identify visible radiographic findings and provide your answer using
ONLY valid JSON.

Use exactly this structure:

{
  "findings": [
    {
      "name": "finding_name",
      "present": true,
      "confidence": 0.0,
      "anatomy": "anatomical_location_or_null"
    }
  ],
  "impression": "short radiographic summary",
  "overall_confidence": 0.0,
  "requires_review": true
}

Rules:

1. confidence must be between 0.0 and 1.0.
2. overall_confidence must be between 0.0 and 1.0.
3. Do not output markdown.
4. Do not output text outside JSON.
5. Do not claim a finding unless it is supported by the image.
6. If no clear abnormality is visible, use an empty findings list.
7. requires_review must remain true because this is an AI research
   system and not a clinical diagnostic tool.
""".strip()


def build_single_image_prompt(
    question: str | None = None,
) -> str:
    """
    Build the chest-X-ray analysis instruction.
    """

    prompt = CHEST_XRAY_ANALYSIS_PROMPT

    if question:
        prompt += (
            "\n\nAdditional research question:\n"
            f"{question.strip()}"
        )

    return prompt