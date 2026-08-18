# Changelog

All notable changes to MedChange-VLM are documented in this file.

The project follows semantic versioning where practical.

---

## [0.1.0] - 2026-08-18

### Added

#### Core longitudinal pipeline
- Longitudinal chest X-ray pair construction from NIH ChestX-ray14-derived data.
- Same-view prior/current pair filtering.
- Temporal state generation for:
  - `absent`
  - `new`
  - `persistent`
  - `resolved`
- Patient-aware dataset splitting and patient-overlap auditing.
- Support for seven target findings:
  - atelectasis
  - cardiomegaly
  - consolidation
  - edema
  - pleural effusion
  - pneumonia
  - pneumothorax

#### BiomedCLIP temporal modeling
- BiomedCLIP biomedical image representation pipeline.
- Prior/current embedding extraction.
- Temporal delta representations.
- Finding-specific temporal classifiers.
- Multi-seed patient-aware temporal evaluation.
- Feature-ablation experiments across prior/current/delta representations.

#### Qwen2.5-VL reasoning
- Qwen2.5-VL-3B-Instruct multimodal temporal reasoning.
- 4-bit quantized local inference.
- Structured temporal output generation.
- Robust JSON extraction and output parsing.
- Multimodal prior/current image reasoning.
- Qwen temporal benchmark scripts.

#### Multimodal fusion
- BiomedCLIP + Qwen evidence fusion.
- Finding-level model agreement detection.
- Model-conflict handling.
- Confidence-aware decision logic.
- Finding-level evidence preservation.
- Unified longitudinal result representation.

#### Safety and uncertainty
- Explicit system-level `uncertain` state.
- Human-review escalation.
- Configurable selective-risk policies:
  - `strict`
  - `change_sensitive`
  - `confidence_margin`
  - `low_confidence_only`
- Configurable safety thresholds.
- Decision-reason tracking.
- Review-flag propagation.
- Selective-risk policy benchmark and tuning.

#### Evidence-grounded reporting
- Structured longitudinal report generation.
- Finding-level disagreement reporting.
- Review-flag explanations.
- Evidence-grounded impression generation.
- Overall-change and uncertainty reporting.

#### FastAPI service
- `/health`
- `/model-info`
- `/runtime-status`
- `/analyze-pair`
- `/cache/clear`
- Uploaded-image validation.
- Request metadata validation.
- Identical-image rejection.
- Structured API error handling.
- Retryable runtime-busy responses.
- Inference runtime management.
- Request-result caching.

#### Streamlit dashboard
- Professional longitudinal analysis interface.
- Prior/current image upload.
- API health display.
- Safety-policy configuration.
- Finding-level temporal results.
- BiomedCLIP/Qwen agreement display.
- Uncertainty and review visualization.
- Evidence-grounded report presentation.
- Validated-scope and research-use disclosures.

#### Evaluation
- Patient-disjoint multi-seed evaluation.
- Finding-level:
  - coverage
  - abstention
  - selective accuracy
  - Macro F1
  - covered error
  - review rate
  - per-state recall
- Pair-level:
  - exact pair match
  - pair abstention
  - pair review rate
  - fully covered pairs
  - mean covered findings
- Selective-risk operating-point analysis.

#### Experimental QLoRA research
- Patient-aware temporal SFT dataset generation.
- Two-image multimodal QLoRA training pipeline.
- 4-bit NF4 training.
- LoRA adaptation of:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- Assistant-only loss masking.
- Change-aware pair sampling.
- Frozen-vs-adapted Qwen evaluation.
- Class-collapse analysis.
- QLoRA retained as experimental and excluded from the default v0.1.0 inference path.

#### Testing
- Unit tests for:
  - core schemas
  - NIH adapters
  - longitudinal pairing
  - temporal modeling
  - model agreement
  - VLM output parsing
  - safety policies
  - API validation
  - runtime busy handling
  - caching
  - QLoRA configuration
  - assistant-only masking
  - change-aware sampling
- End-to-end MedChange smoke testing.

#### Documentation
- Comprehensive project README.
- System architecture documentation.
- Evaluation documentation.
- Safety architecture documentation.
- QLoRA experiment documentation.
- Architecture visualization.

---

### Experimental Findings

The v0.1.0 experiments found that MedChange selective fusion improved reliability among accepted predictions while reducing automated coverage.

Representative patient-aware results:

```text
BiomedCLIP selective accuracy : 0.757 ± 0.023
Qwen selective accuracy       : 0.694 ± 0.046
MedChange selective accuracy  : 0.865 ± 0.012

BiomedCLIP covered error      : 0.243 ± 0.023
Qwen covered error            : 0.306 ± 0.046
MedChange covered error       : 0.135 ± 0.012

MedChange coverage            : 0.709 ± 0.038