from __future__ import annotations

import tempfile
import uuid
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
    get_result_cache,
    get_runtime_manager,
)

from medchange.api.schemas import (
    AnalyzePairResponse,
    CacheClearResponse,
    FindingResponse,
    HealthResponse,
    ModelInfoResponse,
    RuntimeStatusResponse,
)

from medchange.api.service import (
    MedChangeService,
)

from medchange.runtime import (
    RuntimeBusyError,
    build_request_cache_key,
)

from medchange.safety.config import (
    SafetyPolicyConfig,
    VALID_SAFETY_POLICIES,
)

from medchange.safety.validation import (
    validate_longitudinal_pair,
)


# ============================================================
# CONSTANTS
# ============================================================


MAX_UPLOAD_BYTES = (
    10 * 1024 * 1024
)


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================


app = FastAPI(
    title="MedChange-VLM API",
    version=__version__,
    description=(
        "Safety-aware longitudinal chest "
        "X-ray comparison API."
    ),
)


# ============================================================
# CLEAN API ERROR HELPER
# ============================================================


def _api_error(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool = False,
) -> HTTPException:
    """
    Create a consistent API error response.

    Example response:

    {
        "detail": {
            "code": "inference_busy",
            "message": "...",
            "retryable": true
        }
    }
    """

    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    )


# ============================================================
# SERVICE SINGLETON
# ============================================================


@lru_cache(maxsize=1)
def get_service() -> MedChangeService:
    """
    Create one shared MedChangeService.

    BiomedCLIP and Qwen remain lazily loaded inside
    the service instead of being recreated per request.
    """

    return MedChangeService(
        classifier_dir=(
            get_classifier_dir()
        )
    )


# ============================================================
# HEALTH
# ============================================================


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


# ============================================================
# MODEL INFORMATION
# ============================================================


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


# ============================================================
# RUNTIME STATUS
# ============================================================


@app.get(
    "/runtime-status",
    response_model=RuntimeStatusResponse,
)
def runtime_status() -> RuntimeStatusResponse:
    runtime = (
        get_runtime_manager()
    )

    cache = (
        get_result_cache()
    )

    status = (
        runtime.status()
    )

    return RuntimeStatusResponse(
        busy=(
            status.busy
        ),

        total_requests=(
            status.total_requests
        ),

        successful_requests=(
            status.successful_requests
        ),

        failed_requests=(
            status.failed_requests
        ),

        cache_hits=(
            status.cache_hits
        ),

        cache_entries=len(
            cache
        ),

        active_request_id=(
            status.active_request_id
        ),
    )


# ============================================================
# CACHE CONTROL
# ============================================================


@app.post(
    "/cache/clear",
    response_model=CacheClearResponse,
)
def clear_cache() -> CacheClearResponse:
    cache = (
        get_result_cache()
    )

    count = len(
        cache
    )

    cache.clear()

    return CacheClearResponse(
        status="cleared",
        cleared_entries=count,
    )


# ============================================================
# UPLOAD HELPERS
# ============================================================


def _validate_upload_content_type(
    upload: UploadFile,
    *,
    field_name: str,
) -> None:
    """
    Reject unsupported MIME types before writing the file.
    """

    content_type = (
        upload.content_type
        or ""
    )

    if (
        content_type
        not in ALLOWED_CONTENT_TYPES
    ):
        raise _api_error(
            status_code=415,
            code=(
                f"unsupported_{field_name}_type"
            ),
            message=(
                f"{field_name.capitalize()} image "
                "must be PNG or JPEG."
            ),
        )


def _save_upload(
    upload: UploadFile,
    path: Path,
) -> None:
    """
    Save upload with a hard 10 MB size limit.

    The upload is streamed in chunks rather than read
    fully into memory.
    """

    size = 0

    try:
        with path.open(
            "wb"
        ) as output:

            while True:
                chunk = upload.file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(
                    chunk
                )

                if (
                    size
                    > MAX_UPLOAD_BYTES
                ):
                    raise ValueError(
                        "Uploaded image exceeds "
                        "10 MB limit."
                    )

                output.write(
                    chunk
                )

    finally:
        try:
            upload.file.seek(
                0
            )

        except Exception:
            pass


# ============================================================
# RESPONSE CONSTRUCTION
# ============================================================


def _build_response(
    *,
    output: dict,
    safety_config: SafetyPolicyConfig,
    cache_hit: bool,
) -> AnalyzePairResponse:

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

        for item
        in result.findings
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

        cache_hit=(
            cache_hit
        ),
    )


# ============================================================
# ANALYZE PAIR
# ============================================================


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

    # --------------------------------------------------------
    # 1. REQUEST METADATA VALIDATION
    # --------------------------------------------------------

    if not pair_id.strip():
        raise _api_error(
            status_code=422,
            code="invalid_pair_id",
            message=(
                "pair_id must not be empty."
            ),
        )

    if not prior_study_id.strip():
        raise _api_error(
            status_code=422,
            code=(
                "invalid_prior_study_id"
            ),
            message=(
                "prior_study_id must not "
                "be empty."
            ),
        )

    if not current_study_id.strip():
        raise _api_error(
            status_code=422,
            code=(
                "invalid_current_study_id"
            ),
            message=(
                "current_study_id must not "
                "be empty."
            ),
        )

    if (
        prior_study_id.strip()
        == current_study_id.strip()
    ):
        raise _api_error(
            status_code=422,
            code="identical_study_ids",
            message=(
                "Prior and current study IDs "
                "must be different."
            ),
        )

    # --------------------------------------------------------
    # 2. SAFETY POLICY VALIDATION
    # --------------------------------------------------------

    if (
        safety_policy
        not in VALID_SAFETY_POLICIES
    ):
        raise _api_error(
            status_code=422,
            code="invalid_safety_policy",
            message=(
                f"Unsupported safety policy: "
                f"{safety_policy}"
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
        raise _api_error(
            status_code=422,
            code=(
                "invalid_safety_configuration"
            ),
            message=str(
                exc
            ),
        ) from exc

    # --------------------------------------------------------
    # 3. UPLOAD CONTENT-TYPE VALIDATION
    # --------------------------------------------------------

    _validate_upload_content_type(
        prior,
        field_name="prior",
    )

    _validate_upload_content_type(
        current,
        field_name="current",
    )

    # --------------------------------------------------------
    # 4. RESOLVE TEMPORARY FILE EXTENSIONS
    # --------------------------------------------------------

    suffix_prior = (
        Path(
            prior.filename
            or "prior.png"
        )
        .suffix
        .lower()
        or ".png"
    )

    suffix_current = (
        Path(
            current.filename
            or "current.png"
        )
        .suffix
        .lower()
        or ".png"
    )

    # --------------------------------------------------------
    # 5. TEMPORARY WORKSPACE
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # 6. SAVE UPLOADS WITH SIZE GUARD
        # ----------------------------------------------------

        try:
            _save_upload(
                prior,
                prior_path,
            )

            _save_upload(
                current,
                current_path,
            )

        except ValueError as exc:
            raise _api_error(
                status_code=413,
                code="upload_too_large",
                message=str(
                    exc
                ),
            ) from exc

        except OSError as exc:
            raise _api_error(
                status_code=400,
                code="upload_write_failed",
                message=(
                    "Unable to process uploaded "
                    "image files."
                ),
            ) from exc

        # ----------------------------------------------------
        # 7. IMAGE-PAIR VALIDATION
        # ----------------------------------------------------

        try:
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
            raise _api_error(
                status_code=400,
                code="invalid_image_pair",
                message=str(
                    exc
                ),
            ) from exc

        # ----------------------------------------------------
        # 8. CACHE + RUNTIME DEPENDENCIES
        # ----------------------------------------------------

        cache = (
            get_result_cache()
        )

        runtime = (
            get_runtime_manager()
        )

        # ----------------------------------------------------
        # 9. BUILD CACHE KEY
        # ----------------------------------------------------

        cache_key = (
            build_request_cache_key(
                prior_path=(
                    prior_path
                ),

                current_path=(
                    current_path
                ),

                pair_id=(
                    pair_id.strip()
                ),

                prior_study_id=(
                    prior_study_id.strip()
                ),

                current_study_id=(
                    current_study_id.strip()
                ),

                safety_policy=(
                    safety_config.policy
                ),

                safety_threshold=(
                    safety_config.threshold
                ),
            )
        )

        # ----------------------------------------------------
        # 10. CACHE LOOKUP
        # ----------------------------------------------------

        cached = (
            cache.get(
                cache_key
            )
        )

        if cached is not None:

            runtime.record_cache_hit()

            cached_payload = dict(
                cached
            )

            cached_payload[
                "cache_hit"
            ] = True

            return AnalyzePairResponse(
                **cached_payload
            )

        # ----------------------------------------------------
        # 11. REAL GPU INFERENCE
        # ----------------------------------------------------

        request_id = str(
            uuid.uuid4()
        )

        try:
            output = (
                runtime.run(
                    request_id,

                    get_service()
                    .analyze_pair,

                    prior_path=(
                        prior_path
                    ),

                    current_path=(
                        current_path
                    ),

                    pair_id=(
                        pair_id.strip()
                    ),

                    prior_study_id=(
                        prior_study_id.strip()
                    ),

                    current_study_id=(
                        current_study_id.strip()
                    ),

                    safety_config=(
                        safety_config
                    ),
                )
            )

        except RuntimeBusyError as exc:
            raise _api_error(
                status_code=503,
                code="inference_busy",
                message=(
                    "MedChange inference engine "
                    "is currently busy."
                ),
                retryable=True,
            ) from exc

        except Exception as exc:
            # Do NOT expose internal model errors,
            # paths or stack traces to API clients.
            #
            # The original exception remains chained
            # server-side through "from exc".
            raise _api_error(
                status_code=500,
                code="inference_failed",
                message=(
                    "MedChange inference failed."
                ),
                retryable=False,
            ) from exc

        # ----------------------------------------------------
        # 12. BUILD PUBLIC RESPONSE
        # ----------------------------------------------------

        response_payload = (
            _build_response(
                output=(
                    output
                ),

                safety_config=(
                    safety_config
                ),

                cache_hit=False,
            )
        )

        # ----------------------------------------------------
        # 13. CACHE SUCCESSFUL RESPONSE
        # ----------------------------------------------------

        cache.set(
            cache_key,
            response_payload.model_dump(),
        )

        return response_payload