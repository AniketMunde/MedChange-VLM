from __future__ import annotations


CHEST_XRAY_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


def display_finding(
    finding: str,
) -> str:
    return finding.replace(
        "_",
        " ",
    )


def build_positive_prompt(
    finding: str,
) -> str:
    label = display_finding(
        finding
    )

    return (
        "a chest radiograph showing "
        f"{label}"
    )


def build_negative_prompt(
    finding: str,
) -> str:
    label = display_finding(
        finding
    )

    return (
        "a chest radiograph without "
        f"{label}"
    )


def build_prompt_pair(
    finding: str,
) -> tuple[str, str]:

    return (
        build_positive_prompt(
            finding
        ),
        build_negative_prompt(
            finding
        ),
    )