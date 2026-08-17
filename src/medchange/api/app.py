from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from medchange import __version__
from medchange.api.dependencies import (
    TARGET_FINDINGS,
    get_classifier_dir,
    get_default_safety_config,
)
from medchange.api.schemas import (
    AnalyzePairResponse,
    FindingResponse,
    HealthResponse,
    ModelInfoResponse,
)
from medchange.api.service import (
    MedChangeService,
)
from medchange.safety.config import (
    SafetyPolicyConfig,
    VALID_SAFETY_POLICIES,
)
from medchange.safety.validation import (
    validate_longitudinal_pair,
)


app = FastAPI(
    title="MedChange-VLM API",
    version=__version__,
    description=(
        "Safety-aware longitudinal chest "
        "X-ray comparison API."
    ),
)


@lru_cache(maxsize=1)
def get_service() -> MedChangeService:
    return MedChangeService(
        classifier_dir=(
            get_classifier_dir()
        )
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="MedChange-VLM",
        version=__version__,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    safety = (
        get_default_safety_config()
    )

    return ModelInfoResponse(
        biomedclip_model=(
            "microsoft/"
            "BiomedCLIP-PubMedBERT_256-"
            "vit_base_patch16_224"
        ),

        qwen_model=(
            "Qwen/"
            "Qwen2.5-VL-3B-Instruct"
        ),

        qwen_quantization="4-bit",

        safety_policy=(
            safety.policy
        ),

        safety_threshold=(
            safety.threshold
        ),

        target_findings=(
            TARGET_FINDINGS
        ),
    )


def _save_upload(
    upload: UploadFile,
    path: Path,
) -> None:
    with path.open(
        "wb"
    ) as output:
        shutil.copyfileobj(
            upload.file,
            output,
        )


@app.post(
    "/analyze-pair",
    response_model=AnalyzePairResponse,
)
def analyze_pair(
    prior: UploadFile = File(...),
    current: UploadFile = File(...),

    pair_id: str = Form(...),

    prior_study_id: str = Form(...),

    current_study_id: str = Form(...),

    safety_policy: str = Form(
        "change_sensitive"
    ),

    safety_threshold: float | None = Form(
        None
    ),
) -> AnalyzePairResponse:
    if (
        safety_policy
        not in VALID_SAFETY_POLICIES
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported safety policy."
            ),
        )

    try:
        safety_config = (
            SafetyPolicyConfig(
                policy=(
                    safety_policy
                ),

                threshold=(
                    safety_threshold
                ),
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(
                exc
            ),
        ) from exc

    suffix_prior = (
        Path(
            prior.filename
            or "prior.png"
        )
        .suffix
        or ".png"
    )

    suffix_current = (
        Path(
            current.filename
            or "current.png"
        )
        .suffix
        or ".png"
    )

    with tempfile.TemporaryDirectory(
        prefix="medchange_"
    ) as temp_dir:
        temp_dir = Path(
            temp_dir
        )

        prior_path = (
            temp_dir
            / (
                "prior"
                + suffix_prior
            )
        )

        current_path = (
            temp_dir
            / (
                "current"
                + suffix_current
            )
        )

        try:
            _save_upload(
                prior,
                prior_path,
            )

            _save_upload(
                current,
                current_path,
            )

            (
                prior_path,
                current_path,
            ) = (
                validate_longitudinal_pair(
                    prior_path,
                    current_path,
                )
            )

        except (
            FileNotFoundError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc

        try:
            output = (
                get_service()
                .analyze_pair(
                    prior_path=(
                        prior_path
                    ),

                    current_path=(
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

                    safety_config=(
                        safety_config
                    ),
                )
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "MedChange inference failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            ) from exc

    result = (
        output[
            "result"
        ]
    )

    report = (
        output[
            "report"
        ]
    )

    findings = [
        FindingResponse(
            finding=(
                item.finding
            ),

            final_state=(
                item.final_state
            ),

            biomedclip_state=(
                item.biomedclip_state
            ),

            qwen_state=(
                item.qwen_state
            ),

            biomedclip_confidence=(
                item.biomedclip_confidence
            ),

            qwen_confidence=(
                item.qwen_confidence
            ),

            agreement=(
                item.agreement
            ),

            uncertainty=(
                item.uncertainty
            ),

            requires_review=(
                item.requires_review
            ),

            evidence=(
                item.evidence
            ),

            decision_reason=(
                item.decision_reason
            ),
        )

        for item in (
            result.findings
        )
    ]

    return AnalyzePairResponse(
        pair_id=(
            result.pair_id
        ),

        prior_study_id=(
            result.prior_study_id
        ),

        current_study_id=(
            result.current_study_id
        ),

        overall_change=(
            result.overall_change
        ),

        uncertainty=(
            result.uncertainty
        ),

        requires_review=(
            result.requires_review
        ),

        findings=(
            findings
        ),

        impression=(
            report.impression
        ),

        safety_policy=(
            safety_config.policy
        ),

        safety_threshold=(
            safety_config.threshold
        ),

        total_elapsed_seconds=(
            output[
                "total_elapsed_seconds"
            ]
        ),
    )