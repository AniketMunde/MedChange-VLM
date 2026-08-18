# MedChange-VLM Evaluation

## 1. Overview

MedChange-VLM is evaluated as a **longitudinal chest X-ray change-analysis system**, rather than only as a conventional single-image classifier.

The evaluation framework therefore measures several different properties:

- temporal-state classification
- patient-level generalization
- model agreement and disagreement
- selective prediction
- abstention
- review routing
- risk-coverage trade-offs
- pair-level reliability
- experimental QLoRA temporal adaptation

The four underlying temporal states are:

```text
absent
new
persistent
resolved
```

At the final MedChange system level, an additional outcome may appear:

```text
uncertain
```

`uncertain` is not a fifth temporal disease state. It represents a **system-level abstention decision** caused by insufficient or conflicting evidence.

---

# 2. Evaluation Principles

## 2.1 Patient-Aware Splitting

Multiple radiographs from the same patient are statistically related.

Allowing one patient's studies to appear in both training and testing can therefore produce optimistic evaluation results.

MedChange-VLM uses patient-aware splitting for the principal temporal benchmark.

For each evaluated split:

```text
Training patients ∩ Test patients = ∅
```

The M5.5.1 evaluation used five seeds:

```text
11
21
42
84
123
```

Across these runs:

```text
Patient overlap = 0
```

---

## 2.2 Temporal State Evaluation

Each of the seven supported findings is assigned one of four states:

| State | Interpretation |
|---|---|
| `absent` | Finding absent in prior and current studies |
| `new` | Finding appears in current study |
| `persistent` | Finding present in both studies |
| `resolved` | Finding disappears in current study |

The supported findings are:

```text
atelectasis
cardiomegaly
consolidation
edema
pleural_effusion
pneumonia
pneumothorax
```

---

# 3. Metrics

No single metric adequately describes the behavior of a selective longitudinal system.

MedChange-VLM therefore reports several complementary metrics.

## Coverage

Fraction of predictions for which the system returns a temporal state rather than abstaining.

```text
coverage =
number of accepted predictions
/
number of evaluated predictions
```

---

## Abstention Rate

Fraction of predictions routed to uncertainty/review.

```text
abstention rate = 1 - coverage
```

---

## Selective Accuracy

Accuracy calculated only over predictions accepted by the system.

This metric answers:

> When MedChange chooses to make a prediction, how often is that prediction correct?

---

## Covered Error Rate

Error rate among predictions accepted by the system.

```text
covered error rate =
1 - selective accuracy
```

---

## Macro F1

F1 is calculated independently for the temporal states and then averaged.

Macro F1 is particularly important because the temporal dataset is strongly imbalanced toward `absent`.

---

## State Recall

Recall is reported separately for:

```text
absent
new
persistent
resolved
```

This exposes whether apparently strong aggregate performance is actually driven by the dominant state.

---

## Review Rate

Fraction of finding-level predictions explicitly marked as requiring review.

---

# 4. M5.5.1 — Patient-Aware Multi-Seed Benchmark

The principal MedChange fusion benchmark used:

```text
Source pairs : 200
Seeds        : [11, 21, 42, 84, 123]
```

Each seed produced a patient-disjoint train/test split.

The exact number of usable temporal pairs varied slightly between seeds because of the patient-aware split and finding-level training requirements.

Representative split sizes included approximately:

```text
Train pairs : 138–142
Test pairs  : 30–32

Train patients : 133
Test patients  : 29

Patient overlap : 0
```

---

# 5. Finding-Level Benchmark Results

## 5.1 BiomedCLIP

```text
Coverage             : 1.000 ± 0.000
Abstention rate      : 0.000 ± 0.000
Selective accuracy   : 0.757 ± 0.023
Selective Macro F1   : 0.354 ± 0.032
Covered error rate   : 0.243 ± 0.023
Review rate          : 0.000 ± 0.000
```

State recall:

```text
Absent      : 0.902 ± 0.025
New         : 0.172 ± 0.087
Persistent  : 0.117 ± 0.048
Resolved    : 0.198 ± 0.089
```

---

## 5.2 Qwen2.5-VL

```text
Coverage             : 1.000 ± 0.000
Abstention rate      : 0.000 ± 0.000
Selective accuracy   : 0.694 ± 0.046
Selective Macro F1   : 0.269 ± 0.023
Covered error rate   : 0.306 ± 0.046
Review rate          : 0.000 ± 0.000
```

State recall:

```text
Absent      : 0.834 ± 0.049
New         : 0.295 ± 0.122
Persistent  : 0.000 ± 0.000
Resolved    : 0.046 ± 0.054
```

---

## 5.3 MedChange Fusion

```text
Coverage             : 0.709 ± 0.038
Abstention rate      : 0.291 ± 0.038
Selective accuracy   : 0.865 ± 0.012
Selective Macro F1   : 0.275 ± 0.043
Covered error rate   : 0.135 ± 0.012
Review rate          : 0.291 ± 0.038
```

State recall:

```text
Absent      : 0.988 ± 0.011
New         : 0.095 ± 0.104
Persistent  : 0.000 ± 0.000
Resolved    : 0.025 ± 0.056
```

---

# 6. Main Finding-Level Comparison

| System | Coverage | Selective Accuracy | Macro F1 | Covered Error |
|---|---:|---:|---:|---:|
| BiomedCLIP | 1.000 ± 0.000 | 0.757 ± 0.023 | **0.354 ± 0.032** | 0.243 ± 0.023 |
| Qwen2.5-VL | 1.000 ± 0.000 | 0.694 ± 0.046 | 0.269 ± 0.023 | 0.306 ± 0.046 |
| **MedChange** | **0.709 ± 0.038** | **0.865 ± 0.012** | 0.275 ± 0.043 | **0.135 ± 0.012** |

The important result is not that MedChange dominates every metric.

It does not.

BiomedCLIP produced the highest Macro F1 in this experiment.

Instead, the fusion system produced a substantially different operating behavior:

```text
BiomedCLIP covered error : 24.3%
Qwen covered error       : 30.6%
MedChange covered error  : 13.5%
```

while retaining approximately:

```text
70.9% finding-level coverage
```

The improvement in covered accuracy is therefore obtained through **selective abstention**, not through uniformly better classification of every temporal state.

---

# 7. Interpretation of the Fusion Result

The MedChange result demonstrates the intended safety behavior:

```text
Higher reliability on accepted predictions
                    ↑
                    │
                abstention
                    │
                    ↓
Lower automated coverage
```

Compared with the standalone BiomedCLIP model:

```text
Coverage:

BiomedCLIP = 100%
MedChange  ≈ 70.9%
```

but:

```text
Selective accuracy:

BiomedCLIP ≈ 75.7%
MedChange  ≈ 86.5%
```

and:

```text
Covered error:

BiomedCLIP ≈ 24.3%
MedChange  ≈ 13.5%
```

This is the primary selective-risk result of the v0.1 system.

---

# 8. Important Limitation: Temporal Change Recall

The improved covered accuracy does not mean that MedChange solved temporal change detection.

In fact, the state-specific results reveal an important limitation.

MedChange recall was:

```text
Absent      : 98.8%
New         : 9.5%
Persistent  : 0.0%
Resolved    : 2.5%
```

The system is therefore substantially better at confidently identifying stable absence than recognizing less frequent change states.

This is why the evaluation reports state recall alongside accuracy.

A system that predicts or accepts mostly `absent` cases can appear strong under aggregate accuracy while still performing poorly on clinically interesting change states.

---

# 9. Pair-Level Evaluation

Finding-level performance does not fully describe a longitudinal report.

Each pair contains predictions for seven findings.

MedChange therefore also reports pair-level metrics.

M5.5.1 produced:

```text
Exact pair match      : 0.013 ± 0.017
Pair abstention rate  : 0.968 ± 0.023
Pair review rate      : 0.968 ± 0.023
Fully covered pairs   : 0.032 ± 0.023
Mean covered findings : 4.965 ± 0.268
```

Overall:

```text
Conflict rate    : 0.268
Uncertainty rate : 0.290
```

---

# 10. Why Pair-Level Abstention Is High

A pair contains seven findings.

A pair can therefore require review even when most individual findings are accepted.

For example:

```text
Finding 1 → accepted
Finding 2 → accepted
Finding 3 → accepted
Finding 4 → accepted
Finding 5 → accepted
Finding 6 → accepted
Finding 7 → uncertain
```

At finding level:

```text
6 / 7 predictions are covered
```

but at pair level:

```text
the pair requires review
```

This explains why:

```text
finding-level coverage ≈ 70.9%
```

can coexist with:

```text
pair review rate ≈ 96.8%
```

The two metrics answer different questions.

---

# 11. M5.5.2 — Selective-Risk Policy Tuning

Several safety policies were evaluated across confidence thresholds from:

```text
0.55 → 0.80
```

Policies included:

```text
change_sensitive
confidence_margin
low_confidence_only
strict
```

The objective was to study how policy choice changes:

```text
coverage
selective accuracy
macro F1
covered error
change-state recall
```

---

# 12. Change-Sensitive Policy

Results:

| Threshold | Coverage | Accuracy | Macro F1 | Error | New Recall | Persistent Recall | Resolved Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.849 | 0.763 | 0.374 | 0.237 | 0.228 | 0.147 | 0.243 |
| 0.60 | 0.842 | 0.762 | 0.374 | 0.238 | 0.228 | 0.152 | 0.243 |
| 0.65 | 0.836 | 0.761 | 0.375 | 0.239 | 0.235 | 0.152 | 0.243 |
| 0.70 | 0.827 | 0.762 | 0.376 | 0.238 | 0.236 | 0.159 | 0.245 |
| 0.75 | 0.816 | 0.763 | 0.378 | 0.237 | 0.248 | 0.159 | 0.249 |
| 0.80 | 0.806 | 0.769 | **0.383** | 0.231 | **0.265** | 0.159 | **0.266** |

Among the evaluated change-sensitive operating points, threshold `0.80` produced:

```text
Coverage           : 80.6%
Selective accuracy : 76.9%
Macro F1           : 38.3%
New recall         : 26.5%
Persistent recall  : 15.9%
Resolved recall    : 26.6%
```

This policy preserved considerably more change-state recall than the strict fusion configuration.

---

# 13. Confidence-Margin Policy

| Threshold | Coverage | Accuracy | Macro F1 | Error | New Recall | Persistent Recall | Resolved Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.966 | 0.774 | 0.365 | 0.226 | 0.173 | 0.125 | 0.204 |
| 0.60 | 0.947 | 0.779 | 0.362 | 0.221 | 0.173 | 0.109 | 0.195 |
| 0.65 | 0.932 | 0.781 | 0.360 | 0.219 | 0.183 | 0.109 | 0.172 |
| 0.70 | 0.911 | 0.787 | 0.361 | 0.213 | 0.177 | 0.112 | 0.163 |
| 0.75 | 0.889 | 0.792 | 0.342 | 0.208 | 0.186 | 0.077 | 0.138 |
| 0.80 | 0.866 | **0.805** | 0.339 | **0.195** | 0.192 | 0.043 | 0.149 |

At threshold `0.80`:

```text
Coverage           ≈ 86.6%
Selective accuracy ≈ 80.5%
Covered error      ≈ 19.5%
```

This provides a less conservative operating point than the strict policy.

---

# 14. Low-Confidence-Only Policy

In the recorded M5.5.2 experiment, `low_confidence_only` produced the same numerical operating points as `confidence_margin`.

For example, at threshold `0.80`:

```text
Coverage           : 0.866
Selective accuracy : 0.805
Macro F1           : 0.339
Covered error      : 0.195
```

This should be interpreted as an empirical result of the evaluated implementation/data rather than evidence that the two policy definitions are universally equivalent.

---

# 15. Strict Policy

| Threshold | Coverage | Accuracy | Macro F1 | Error | New Recall | Persistent Recall | Resolved Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.716 | 0.865 | 0.275 | 0.135 | 0.095 | 0.000 | 0.025 |
| 0.60 | 0.709 | 0.865 | 0.275 | 0.135 | 0.095 | 0.000 | 0.025 |
| 0.65 | 0.703 | 0.865 | 0.276 | 0.135 | 0.099 | 0.000 | 0.025 |
| 0.70 | 0.692 | 0.867 | 0.259 | 0.133 | 0.079 | 0.000 | 0.000 |
| 0.75 | 0.680 | 0.871 | 0.262 | 0.129 | 0.090 | 0.000 | 0.000 |
| 0.80 | 0.669 | **0.882** | 0.265 | **0.118** | 0.090 | 0.000 | 0.000 |

The most conservative tested operating point was:

```text
Policy    : strict
Threshold : 0.80

Coverage           : 66.9%
Selective accuracy : 88.2%
Covered error      : 11.8%
```

This result illustrates the central risk-coverage trade-off:

```text
more abstention
     ↓
lower coverage
     ↓
lower error among accepted predictions
```

---

# 16. Policy Trade-Off

No single policy is optimal for every objective.

### Strict

Prioritizes:

```text
lower covered error
higher selective accuracy
```

at the cost of:

```text
lower coverage
very weak change-state recall
```

### Change-Sensitive

Provides stronger observed recall for:

```text
new
persistent
resolved
```

but accepts more prediction risk.

### Confidence-Margin / Low-Confidence-Only

Provide higher coverage and intermediate selective accuracy.

Therefore, policy selection should be treated as an **operating-point decision**, not as a universal ranking.

---

# 17. Example Safety-Aware Inference

A representative MedChange inference produced:

```text
atelectasis:

BiomedCLIP = absent
Qwen       = new
```

with:

```text
agreement   = conflict
uncertainty = high
```

Using:

```text
Safety policy : change_sensitive
Threshold     : 0.80
```

the final result became:

```text
Final state     : uncertain
Requires review : True
```

and the report explained that the configured safety policy abstained because the BiomedCLIP prediction conflicted with Qwen.

This is the intended behavior of the safety layer.

---

# 18. Experimental M7 QLoRA Evaluation

QLoRA adaptation was investigated after completion of the core MedChange fusion system.

These experiments are reported separately because the resulting adapters are **not used in the default v0.1.0 inference pipeline**.

---

# 19. Initial M7 Dataset

The initial temporal SFT smoke dataset contained:

```text
Train
Samples  : 426
Patients : 74

Validation
Samples  : 59
Patients : 9

Test
Samples  : 15
Patients : 10

Patient overlap : 0
```

The initial test split was extremely small and strongly imbalanced.

Across seven findings and 15 pairs:

```text
absent      : 98
resolved    : 4
persistent  : 2
new         : 1
```

This made it unsuitable for drawing strong conclusions about temporal adaptation.

---

# 20. Initial QLoRA Training Behavior

A 20-step QLoRA experiment showed decreasing optimization loss.

Training loss progressed approximately from:

```text
8.099
```

to:

```text
4.755
```

with final:

```text
train_loss : 6.096
eval_loss  : 4.761
```

However, downstream temporal evaluation remained extremely poor.

Base Qwen:

```text
Accuracy              : 0.0286
Macro F1              : 0.0188
Exact pair match      : 0.000
Parse success         : 1.000
```

QLoRA-adapted Qwen:

```text
Accuracy              : 0.0286
Macro F1              : 0.0163
Exact pair match      : 0.000
Parse success         : 1.000
```

The initial adapter therefore did not improve the temporal task.

---

# 21. M7.5 — Larger Change-Aware Dataset

To improve the experiment, a larger patient-aware dataset was created.

Source population:

```text
Source pairs    : 61,219
Source patients : 12,051
Patient overlap : 0
```

Final selected splits:

| Split | Samples | Patients |
|---|---:|---:|
| Train | 3,000 | 1,975 |
| Validation | 400 | 260 |
| Test | 500 | 318 |

---

# 22. M7.5 Training State Distribution

Training finding states:

```text
absent      : 18,053
new         : 1,132
persistent  : 705
resolved    : 1,110
```

Total finding-level labels:

```text
21,000
```

because:

```text
3,000 pairs × 7 findings = 21,000 labels
```

Although change-containing pairs were deliberately sampled, `absent` remained the dominant finding-level state.

---

# 23. M7.5 Pair Categories

Training pair categories were:

```text
persistent : 600
new        : 750
resolved   : 600
stable     : 1,050
```

Validation:

```text
persistent : 66
new        : 80
resolved   : 67
stable     : 187
```

Test:

```text
persistent : 96
new        : 89
resolved   : 62
stable     : 253
```

This represented a substantial improvement over the original 15-pair test split.

---

# 24. M7.5 QLoRA Configuration

The experimental adapter used:

```text
Base model:
Qwen2.5-VL-3B-Instruct

Quantization:
4-bit

LoRA rank:
8

LoRA alpha:
16

LoRA dropout:
0.05
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
```

Trainable parameters:

```text
3,686,400
```

Total parameters:

```text
3,758,309,376
```

Trainable fraction:

```text
0.0981%
```

---

# 25. M7.5 Smoke Training

The experiment used:

```text
max_steps                    : 20
per_device_train_batch_size  : 1
gradient_accumulation_steps  : 4
learning_rate                : 2e-4
fp16                         : True
gradient_checkpointing       : True
optimizer                    : paged_adamw_8bit
```

Training loss decreased substantially.

Examples:

```text
Step 1  : 0.3160
Step 5  : 0.1359
Step 10 : 0.0706
Step 15 : 0.0808
Step 20 : 0.0400
```

Validation loss also decreased:

```text
Step 5  : 0.1347
Step 10 : 0.0788
Step 15 : 0.0635
Step 20 : 0.0597
```

Final validation loss:

```text
0.059724
```

At optimization level, the adaptation therefore appeared successful.

---

# 26. M7.5 Held-Out Temporal Evaluation

However, downstream temporal evaluation showed a different result.

On a 100-pair evaluation subset:

```text
Accuracy              : 0.8914
Macro F1              : 0.2356
Exact pair match rate : 0.530
Parse success rate    : 1.000
```

State recall:

```text
Absent      : 1.000
New         : 0.000
Persistent  : 0.000
Resolved    : 0.000
```

The adapter had collapsed toward the dominant `absent` state.

---

# 27. Why 89.1% Accuracy Is Misleading

At first glance:

```text
Accuracy = 89.1%
```

looks strong.

However:

```text
New recall        = 0%
Persistent recall = 0%
Resolved recall   = 0%
```

The model failed to identify every evaluated non-absent temporal state.

Therefore, the correct interpretation is:

> The adapter learned a high-accuracy majority-state solution rather than robust longitudinal change reasoning.

This is one of the most important negative findings from the M7 experiments.

---

# 28. Training Loss Is Not Task Performance

The QLoRA experiment demonstrates why optimization loss alone cannot validate a medical temporal model.

The experiment produced:

```text
Validation LM loss ≈ 0.060
```

while simultaneously producing:

```text
New recall        = 0
Persistent recall = 0
Resolved recall   = 0
```

Therefore:

```text
low language-model loss
        ≠
strong temporal discrimination
```

Downstream task-specific evaluation remains necessary.

---

# 29. Why QLoRA Is Not Enabled in v0.1.0

The experimental adapter was not integrated into the default inference pipeline because it did not demonstrate reliable change-state recognition.

The default architecture therefore continues to use the original Qwen pathway together with BiomedCLIP, fusion, and safety-aware abstention.

QLoRA remains:

```text
experimental
```

rather than:

```text
production/default
```

within the project.

---

# 30. Evaluation Summary

The current experiments support several conclusions.

## Finding 1 — Patient separation works

The main evaluation uses patient-disjoint splits with:

```text
patient overlap = 0
```

---

## Finding 2 — BiomedCLIP remains the strongest standalone model by Macro F1

In M5.5.1:

```text
BiomedCLIP Macro F1 = 0.354
Qwen Macro F1       = 0.269
MedChange Macro F1  = 0.275
```

The fusion system should therefore not be described as universally outperforming the component models.

---

## Finding 3 — Fusion improves reliability on covered predictions

MedChange reduced covered error from:

```text
BiomedCLIP : 24.3%
Qwen       : 30.6%
```

to:

```text
MedChange : 13.5%
```

while retaining approximately:

```text
70.9% finding-level coverage
```

---

## Finding 4 — Abstention is responsible for much of the reliability improvement

MedChange does not obtain higher selective accuracy for free.

It abstains on approximately:

```text
29.1%
```

of finding-level decisions under the evaluated configuration.

---

## Finding 5 — Pair-level review remains very high

Because one uncertain finding can flag an entire seven-finding pair:

```text
Pair review rate ≈ 96.8%
```

This remains a major usability limitation.

---

## Finding 6 — Change-state recognition remains difficult

Across the current experiments, `new`, `persistent`, and `resolved` remain substantially more difficult than `absent`.

---

## Finding 7 — Safety policy changes the operating point

The evaluated policies expose a measurable trade-off between:

```text
coverage
accuracy
error
change-state recall
```

rather than producing one universally optimal configuration.

---

## Finding 8 — QLoRA requires further work

The M7.5 experiment demonstrated that:

```text
good optimization behavior
```

does not necessarily produce:

```text
good temporal reasoning
```

The adapter collapsed toward `absent` and is therefore not used in the final v0.1.0 inference pipeline.

---

# 31. Limitations of the Current Evaluation

The current evaluation has several important limitations.

### Limited dataset source

The experiments are primarily based on NIH ChestX-ray14-derived longitudinal data.

### Derived temporal supervision

Temporal states are derived from dataset labels rather than expert-authored longitudinal comparison annotations.

### Class imbalance

`absent` strongly dominates finding-level labels.

### Small fusion benchmark

The principal M5.5.1 fusion experiment uses a relatively small 200-pair source benchmark.

### Weak change-state recall

Performance for `new`, `persistent`, and `resolved` remains limited.

### High pair-level review rate

The current safety policy frequently escalates complete pairs.

### General-purpose VLM

Qwen2.5-VL is not a clinically validated radiology model.

### No external validation

The current results do not establish generalization to independent institutions or datasets.

### No radiologist comparison

The system has not yet been evaluated prospectively against expert longitudinal interpretations.

### Experimental QLoRA evaluation size

The reported M7.5 adapter result was obtained from a 100-pair evaluation subset rather than the complete 500-pair held-out split.

---

# 32. Future Evaluation Priorities

The next evaluation phase should prioritize:

```text
larger patient-disjoint benchmarks
        ↓
better non-absent state representation
        ↓
per-finding state-balanced evaluation
        ↓
confidence calibration
        ↓
risk-coverage curves
        ↓
external dataset validation
        ↓
radiologist comparison
        ↓
spatial / lesion grounding evaluation
```

For QLoRA specifically, future work should investigate:

```text
state-balanced sampling
per-finding balancing
loss reweighting
hard-example sampling
longer controlled training
better temporal prompts
larger held-out evaluation
adapter checkpoint comparison
```

---

# 33. Reproducing Evaluation

The repository contains scripts for the major evaluation stages.

Examples include temporal benchmark construction, Qwen evaluation, fusion evaluation, selective-risk analysis, and QLoRA evaluation.

Run the complete unit test suite before reproducing experiments:

```bash
python -m pytest -v
```

Exact experiment commands depend on local dataset paths and model availability.

Generated datasets, downloaded model weights, and QLoRA checkpoints are intentionally excluded from version control.

---

# 34. Final Interpretation

The strongest conclusion from the current MedChange-VLM experiments is not:

> "The fusion model solves longitudinal chest X-ray interpretation."

The results do not support that claim.

Instead, the experiments demonstrate that independent multimodal predictions can be combined with explicit disagreement detection and selective abstention to reduce the error rate among predictions the system chooses to accept.

At the same time, the evaluation exposes significant unresolved limitations in change-state recall, pair-level review burden, dataset imbalance, and temporal adaptation.

That combination of positive and negative evidence defines the current MedChange-VLM v0.1.0 research result.