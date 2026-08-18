# MedChange-VLM Safety Architecture

## 1. Purpose

MedChange-VLM is designed as a **research prototype for longitudinal chest X-ray analysis**.

Its safety architecture is based on a simple principle:

> When the available evidence is conflicting, insufficient, malformed, or outside the validated scope, the system should not silently force a confident conclusion.

Safety is implemented across several layers:

```text
Input validation
      ↓
Structured model outputs
      ↓
Model agreement analysis
      ↓
Uncertainty estimation
      ↓
Selective-risk policy
      ↓
Abstention / review routing
      ↓
Evidence-grounded reporting
      ↓
API/runtime safeguards
```

The system is not clinically validated and does not provide medical-device-grade guarantees.

---

# 2. Safety Boundary

MedChange-VLM v0.1.0 is restricted to seven target findings:

```text
atelectasis
cardiomegaly
consolidation
edema
pleural_effusion
pneumonia
pneumothorax
```

The system should not be interpreted as supporting arbitrary chest X-ray abnormalities.

Its validated research behavior is limited to longitudinal reasoning over this predefined set.

---

# 3. Research-Only Status

MedChange-VLM is:

```text
a research prototype
```

and is not:

```text
a medical device
a diagnostic system
a treatment recommendation system
a radiologist replacement
a clinically validated decision-support tool
```

The system must not be used for patient-care decisions.

---

# 4. Input Safety

Before multimodal inference begins, the system validates the incoming longitudinal pair.

Checks include:

- input file existence
- image decoding
- supported file/content type
- image dimensions
- valid prior/current pair structure
- request metadata validation
- identical image detection
- supported safety-policy configuration

If a validation check fails:

```text
inference does not proceed
```

This prevents invalid or unsupported input from entering the inference pipeline.

---

# 5. Identical-Image Guard

A longitudinal comparison requires two distinct studies.

If the prior and current files contain identical content, the request is rejected.

Conceptually:

```text
Prior image bytes
        =
Current image bytes
        ↓
Invalid longitudinal pair
        ↓
Reject request
```

This avoids meaningless temporal analysis of the same image against itself.

---

# 6. Request Validation

The API validates both uploaded images and request metadata.

Examples include:

```text
pair_id
prior_study_id
current_study_id
safety_policy
safety_threshold
```

Invalid configuration is returned as a structured client error instead of being silently corrected.

---

# 7. Structured VLM Output Safety

Qwen2.5-VL generates language-model output.

Raw language-model text is never treated as a trusted final result.

The pipeline performs:

```text
Raw response
     ↓
JSON extraction
     ↓
Schema validation
     ↓
Finding validation
     ↓
Temporal-state normalization
     ↓
Structured model evidence
```

Malformed or unsupported outputs do not silently propagate into the final report.

---

# 8. Independent Evidence Streams

MedChange-VLM uses two independent evidence pathways:

```text
BiomedCLIP temporal classifier
              +
Qwen2.5-VL temporal reasoning
```

Neither model is automatically treated as the final authority.

This enables the system to detect disagreement explicitly.

Example:

```text
BiomedCLIP : absent
Qwen       : new
```

Rather than choosing one prediction invisibly, the fusion layer records:

```text
agreement = conflict
```

---

# 9. Agreement States

The fusion layer distinguishes between several forms of evidence relationship.

## Agreement

Example:

```text
BiomedCLIP = absent
Qwen       = absent
```

The models support the same temporal state.

---

## Conflict

Example:

```text
BiomedCLIP = absent
Qwen       = new
```

The models disagree on the temporal interpretation.

---

## Uncertain

A finding can become uncertain when:

```text
model disagreement exists
confidence is insufficient
safety-policy requirements are not satisfied
```

The system does not treat uncertainty as a software error.

---

# 10. Uncertainty Is a System-Level Decision

The core temporal states are:

```text
absent
new
persistent
resolved
```

`uncertain` is different.

It means:

> The current system does not consider the available evidence sufficient to safely commit to one of the temporal states.

Therefore:

```text
uncertain != fifth disease state
```

Instead:

```text
uncertain = abstention decision
```

---

# 11. Human Review Flag

When a finding is uncertain or conflicts with the configured safety policy, the system can set:

```text
requires_review = True
```

A review flag can be triggered by:

- model conflict
- insufficient confidence
- change-sensitive disagreement
- safety-policy abstention
- unresolved uncertainty

This status is preserved through:

```text
fusion result
      ↓
report
      ↓
API response
      ↓
Streamlit dashboard
```

---

# 12. Selective Prediction

MedChange-VLM is designed to support selective prediction.

A conventional classifier often behaves like:

```text
input
  ↓
forced class prediction
```

MedChange instead allows:

```text
input
  ↓
evidence evaluation
  ↓
accept OR abstain
```

This means the system can deliberately reduce automated coverage in exchange for lower error among accepted predictions.

---

# 13. Supported Safety Policies

The research implementation includes several configurable safety policies:

```text
strict
change_sensitive
confidence_margin
low_confidence_only
```

These policies define different operating points between:

```text
coverage
risk
change sensitivity
abstention
```

They should be interpreted as experimental policy configurations, not clinically validated operating modes.

---

# 14. Strict Policy

The `strict` policy is the most conservative evaluated configuration.

Its behavior prioritizes:

```text
agreement
higher confidence
lower covered error
```

at the cost of:

```text
lower coverage
more abstention
```

In the M5.5.2 experiments, stricter thresholds produced the lowest covered error among evaluated operating points.

---

# 15. Change-Sensitive Policy

The `change_sensitive` policy is designed for longitudinal tasks where disagreement about a change may deserve more caution.

Change states include:

```text
new
persistent
resolved
```

Example:

```text
BiomedCLIP = absent
Qwen       = new
```

Under a change-sensitive configuration, this disagreement can trigger abstention.

Representative decision:

```text
Final state     : uncertain
Requires review : True
```

This avoids automatically accepting a potentially unsupported new abnormality.

---

# 16. Confidence-Margin Policy

The `confidence_margin` policy evaluates whether available model confidence is sufficient to support the selected output.

Conceptually:

```text
prediction
    +
confidence
    ↓
threshold check
    ↓
accept / abstain
```

Higher thresholds generally increase abstention.

---

# 17. Low-Confidence-Only Policy

The `low_confidence_only` configuration focuses primarily on confidence-based abstention.

In the recorded M5.5.2 experiment, its numerical results matched the confidence-margin policy.

This should not be interpreted as proof that the two policies are theoretically identical.

---

# 18. Safety Threshold

Policies may use a configurable threshold.

Example:

```text
Safety policy : change_sensitive
Threshold     : 0.80
```

Increasing the threshold can reduce coverage while increasing reliability among accepted predictions.

The threshold is therefore an operating-point parameter.

It is not a calibrated clinical probability threshold.

---

# 19. Decision Reason

Every fused finding can include a human-readable decision explanation.

Example:

```text
BiomedCLIP and Qwen produced conflicting temporal states.
Configured safety policy 'change_sensitive'
(threshold=0.80): absent prediction conflicts with Qwen;
policy abstained.
```

This is preserved separately from the final temporal state.

The objective is to make safety decisions inspectable.

---

# 20. Evidence Preservation

When Qwen provides textual evidence, MedChange preserves it alongside the structured finding.

Example:

```text
The current image shows a more prominent opacity
in the left lung field compared to the prior image.
```

The evidence is not treated as proof of correctness.

Instead, it provides context for:

```text
model reasoning
review
debugging
research evaluation
```

---

# 21. Evidence-Grounded Reporting

The report generator receives the fused structured result.

It does not independently generate a new clinical interpretation from scratch.

The reporting pipeline is:

```text
Structured fused findings
          ↓
Safety/review state
          ↓
Evidence
          ↓
Deterministic report construction
```

This reduces the risk that a final free-form language model introduces unsupported claims after the safety decision has already been made.

---

# 22. Review-Focused Report Structure

When uncertainty is present, the report explicitly exposes it.

Example:

```text
Overall change : uncertain
Uncertainty    : high
Review needed  : True
```

It then lists:

```text
uncertain findings
conflicting model predictions
review reasons
supporting evidence
```

The report is intentionally transparent about disagreement.

---

# 23. Overall Change Safety

The overall pair-level result is derived from the finding-level fused outputs.

If a clinically relevant finding remains unresolved or requires review, the overall result may become:

```text
uncertain
```

This avoids presenting a pair as confidently stable when one of its component findings has unresolved disagreement.

---

# 24. Pair-Level Review

A longitudinal pair can contain seven finding-level predictions.

Even if six findings are accepted, one uncertain finding can trigger:

```text
pair review required
```

This is why pair-level review rates can be substantially higher than finding-level abstention rates.

---

# 25. API Error Model

The API uses structured error responses for expected failure conditions.

Conceptually:

```json
{
  "code": "inference_busy",
  "message": "MedChange inference engine is currently busy.",
  "retryable": true
}
```

This separates:

```text
client input errors
runtime availability
internal inference failure
```

instead of returning indistinguishable generic exceptions.

---

# 26. Validation Errors

Invalid requests can return client-side errors such as:

```text
400
422
```

Examples include:

```text
invalid image pair
identical file content
unsupported safety policy
invalid threshold
invalid request data
```

These should not be converted into generic internal server errors.

---

# 27. Runtime Busy Protection

Qwen2.5-VL and BiomedCLIP inference can consume significant GPU resources.

The runtime manager prevents uncontrolled simultaneous inference.

Conceptually:

```text
Request A
   ↓
GPU inference active

Request B
   ↓
runtime busy
   ↓
503 retryable response
```

This protects system stability in the local research deployment.

---

# 28. Runtime Status

The API exposes runtime state information including values such as:

```text
busy
total_requests
successful_requests
failed_requests
cache_hits
cache_entries
active_request_id
```

This allows the dashboard or developer to distinguish:

```text
model failure
```

from:

```text
runtime currently occupied
```

---

# 29. Request Caching

Identical repeated requests can reuse a cached result.

The cache key incorporates relevant request content and configuration.

This prevents repeated expensive inference when:

```text
same images
same study IDs
same safety settings
```

are analyzed again.

---

# 30. Cache Safety

Caching must not cause results from one configuration to be incorrectly reused for another.

Therefore, the request key must distinguish relevant inputs including safety configuration.

Conceptually:

```text
Image pair A + threshold 0.80
```

must not be treated as equivalent to:

```text
Image pair A + threshold 0.60
```

if the safety decision can differ.

---

# 31. Cache Clearing

The API includes a cache-clear endpoint for research/development workflows.

This is useful when:

```text
models change
safety logic changes
development configuration changes
```

and previously cached outputs should no longer be reused.

---

# 32. API Health Endpoint

The system exposes:

```text
GET /health
```

A healthy response indicates that the API service itself is running.

It does not constitute clinical validation or guarantee that every model dependency is currently capable of completing inference.

---

# 33. Model Information Endpoint

The API exposes model/configuration information through:

```text
GET /model-info
```

This provides transparency about:

```text
BiomedCLIP model
Qwen model
Qwen quantization
default safety policy
default threshold
supported findings
```

The goal is to make the runtime configuration inspectable.

---

# 34. Dashboard Safety Disclosure

The Streamlit dashboard displays an explicit scope warning.

Representative disclosure:

```text
Validated scope: longitudinal comparison of seven target
findings — atelectasis, cardiomegaly, consolidation, edema,
pleural effusion, pneumonia, and pneumothorax.

Research prototype only.
```

This prevents the interface from visually implying general diagnostic capability.

---

# 35. Unsupported Findings

If an image contains an abnormality outside the seven supported findings, MedChange-VLM does not provide validated behavior for that abnormality.

For example:

```text
rib fracture
mass
nodule
device complication
mediastinal abnormality
other pathology
```

may be visible in an input image but are outside the current system scope.

The system should not be interpreted as having ruled them out simply because they are absent from the MedChange output.

---

# 36. Random or Non-Chest Images

Input validation reduces accidental misuse, but the current research prototype should not be assumed to provide perfect out-of-distribution detection.

A random image passing basic file validation does not imply that the resulting model output is meaningful.

Future work should include stronger:

```text
modality validation
anatomical validation
out-of-distribution detection
```

before inference.

---

# 37. QLoRA Safety Decision

Experimental QLoRA adapters are not enabled in the default inference pipeline.

The M7 experiments produced:

```text
good optimization loss
```

but poor temporal change-state behavior.

The assistant-only QLoRA experiment collapsed toward:

```text
absent
```

with:

```text
New recall        : 0
Persistent recall : 0
Resolved recall   : 0
```

Therefore the adapter was not promoted into the default system.

This is itself a safety decision:

> A model should not be integrated because its training loss looks good if downstream task behavior is unreliable.

---

# 38. What the Safety Layer Does Not Guarantee

The safety system does not guarantee:

```text
clinical correctness
absence of hallucination
radiological validity
perfect uncertainty calibration
perfect conflict detection
generalization to unseen hospitals
generalization to unsupported diseases
```

It only provides additional safeguards around the current experimental model pipeline.

---

# 39. Known Safety Limitations

Important current limitations include:

### Weak change-state recall

`new`, `persistent`, and `resolved` remain substantially more difficult than `absent`.

### High pair-level review rate

The current policy may escalate a large fraction of complete pairs.

### General-purpose VLM

Qwen2.5-VL is not a clinically validated radiology model.

### Dataset-derived labels

Temporal supervision is derived from available NIH labels rather than expert longitudinal reports.

### Limited scope

Only seven findings are supported.

### No external validation

The current system has not been validated across institutions.

### No prospective evaluation

There has been no prospective clinical deployment or expert workflow study.

---

# 40. Safety-Performance Trade-Off

The safety layer exposes a fundamental trade-off:

```text
higher coverage
      ↑
      │
      │
      │
      ↓
lower abstention
```

versus:

```text
lower covered error
      ↑
      │
      │
      │
      ↓
more abstention
```

The strict policy demonstrated that stronger abstention can improve selective accuracy, but it also reduces coverage and change-state recall.

There is no universally optimal operating point.

---

# 41. Why Accuracy Alone Is Unsafe

A strongly imbalanced temporal dataset can make majority-class predictions appear successful.

The M7.5 QLoRA experiment achieved approximately:

```text
Accuracy : 89.1%
```

while:

```text
New recall        : 0%
Persistent recall : 0%
Resolved recall   : 0%
```

Therefore, safety evaluation must inspect:

```text
per-state recall
macro F1
coverage
covered error
abstention
```

rather than relying on aggregate accuracy.

---

# 42. Human Oversight Principle

The intended research architecture follows:

```text
AI evidence
    ↓
structured uncertainty
    ↓
review routing
    ↓
human oversight
```

rather than:

```text
AI output
    ↓
automatic clinical decision
```

The current system does not attempt to remove human oversight from uncertain cases.

---

# 43. Planned Safety Improvements

Future work should investigate:

```text
stronger modality validation
out-of-distribution detection
calibrated uncertainty
per-finding calibration
external validation
lesion grounding
expert review
balanced temporal adaptation
agentic evidence verification
```

---

# 44. Future Agentic Verification

A future agentic verification layer may include:

```text
initial prediction
      ↓
evidence verifier
      ↓
conflict detector
      ↓
targeted re-analysis
      ↓
adjudication
      ↓
accept / escalate
```

Such a layer should remain constrained by deterministic safety rules.

Agentic reasoning should not be allowed to bypass:

```text
scope restrictions
review requirements
input validation
human oversight
```

---

# 45. Clinical Deployment Boundary

Before any clinical use, substantial additional work would be required, including:

```text
external validation
radiologist evaluation
prospective studies
calibration studies
data governance review
regulatory assessment
clinical workflow testing
security review
medical-device compliance
```

None of these are completed in v0.1.0.

---

# 46. Safety Summary

MedChange-VLM implements safety through multiple independent mechanisms:

```text
validated scope
      +
input validation
      +
structured VLM parsing
      +
independent evidence pathways
      +
agreement analysis
      +
uncertainty
      +
selective prediction
      +
abstention
      +
review flags
      +
evidence-grounded reporting
      +
runtime safeguards
```

The purpose is not to claim that the system is clinically safe.

The purpose is to make uncertainty, disagreement, and system limitations explicit rather than hiding them behind a single automated prediction.

---

# 47. Final Safety Statement

MedChange-VLM is a research prototype.

Any output should be interpreted as:

```text
experimental model evidence
```

not:

```text
medical diagnosis
```

When the system reports uncertainty or disagreement, the appropriate interpretation is that the current model evidence is insufficient for an automated conclusion.

The presence of a confident prediction does not remove the requirement for appropriate expert interpretation in any real clinical setting.