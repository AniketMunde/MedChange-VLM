# MedChange-VLM QLoRA Experiments

## 1. Overview

MedChange-VLM includes an experimental parameter-efficient fine-tuning workflow for adapting Qwen2.5-VL to longitudinal chest X-ray reasoning.

The objective was to test whether QLoRA could improve recognition of the four temporal states used by the project:

```text
absent
new
persistent
resolved
```

The experimental pipeline explored:

- patient-disjoint temporal SFT construction
- two-image multimodal supervision
- 4-bit model quantization
- QLoRA adaptation
- LoRA on attention projection layers
- assistant-only loss masking
- change-aware pair sampling
- held-out temporal evaluation

The resulting adapter is **not enabled in the default MedChange-VLM v0.1.0 inference pipeline**.

The reason is empirical:

> The training pipeline worked correctly, but the adapted model did not demonstrate reliable recognition of non-absent temporal states.

---

# 2. Research Question

The main research question was:

> Can parameter-efficient adaptation of Qwen2.5-VL improve longitudinal chest X-ray temporal reasoning without requiring full-model fine-tuning?

The desired output for each finding was one of:

```text
absent
new
persistent
resolved
```

The adaptation was intended to improve the Qwen branch of MedChange-VLM while keeping the existing multimodal architecture intact.

---

# 3. Base Model

The experiments used:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

The model was loaded with 4-bit quantization to reduce GPU memory requirements.

The experiments were designed for resource-constrained local training rather than large-scale full-parameter fine-tuning.

---

# 4. Why QLoRA

Full fine-tuning of a multi-billion-parameter VLM is computationally expensive.

QLoRA allows the base model to remain quantized while training a comparatively small set of low-rank adapter parameters.

The adaptation strategy used:

```text
4-bit NF4 quantization
        +
frozen base model
        +
LoRA adapters
```

This made it possible to train the 3B model locally using a consumer GPU.

---

# 5. LoRA Configuration

The experimental LoRA configuration used:

```text
LoRA rank    : 8
LoRA alpha   : 16
LoRA dropout : 0.05
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
```

The resulting parameter counts were:

```text
Trainable parameters : 3,686,400
Total parameters     : 3,758,309,376
Trainable fraction   : 0.0981%
```

Therefore, less than 0.1% of the total model parameters were optimized.

---

# 6. Quantization

The model was loaded using 4-bit quantization.

The experimental setup used:

```text
load_in_4bit              : True
quantization type         : NF4
double quantization       : True
compute dtype             : float16
```

The quantized base model was prepared for k-bit training before attaching the LoRA adapters.

---

# 7. Initial Temporal SFT Dataset

The first QLoRA experiment used a small smoke-test dataset derived from the longitudinal same-view temporal pairs.

The split contained:

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
```

Patient overlap:

```text
0
```

The patient-disjoint split was considered essential even for smoke testing.

---

# 8. Initial Test Distribution

The 15-pair test split contained:

```text
15 pairs × 7 findings = 105 finding-level labels
```

Distribution:

```text
absent      : 98
resolved    : 4
persistent  : 2
new         : 1
```

This was extremely imbalanced.

The split was therefore useful as a functional test but insufficient for strong conclusions about temporal-state performance.

---

# 9. M7.2 — Two-Step QLoRA Smoke Test

The first objective was simply to verify that the complete QLoRA training pipeline worked.

The test validated:

```text
4-bit Qwen loading
        ↓
LoRA adapter attachment
        ↓
multimodal two-image batches
        ↓
gradient flow
        ↓
gradient checkpointing
        ↓
validation
        ↓
adapter checkpoint saving
```

The two-step experiment produced:

```text
Step 1 loss : 8.099
Step 2 loss : 7.824

Eval loss   : 7.748
Train loss  : 7.961
```

The training run completed successfully.

This established that the QLoRA infrastructure was technically functional.

---

# 10. M7.3 — 20-Step Naive QLoRA Run

The next experiment extended the same setup to 20 optimizer steps.

The training objective at this stage used the full conversational sequence as labels.

This meant that the loss included:

```text
system prompt
user prompt
multimodal conversation structure
assistant response
```

rather than focusing only on the temporal target.

---

# 11. M7.3 Training Loss

Training loss decreased throughout the run.

Representative values:

```text
Step 1  : 8.099
Step 5  : 7.109
Step 10 : 6.056
Step 15 : 5.107
Step 20 : 4.755
```

Final reported train loss:

```text
6.0962
```

---

# 12. M7.3 Validation Loss

Validation loss also decreased consistently:

```text
Step 5  : 6.934
Step 10 : 5.833
Step 15 : 5.026
Step 20 : 4.761
```

Final validation loss:

```text
4.7613
```

At this point, the optimization behavior appeared healthy.

However, the next experiment showed that the improved loss did not correspond to improved temporal reasoning.

---

# 13. M7.4 — Frozen Qwen vs Naive QLoRA

The base model and M7.3 adapter were evaluated on the 15-pair held-out test split.

## Base Qwen

```text
Accuracy              : 0.0286
Macro F1              : 0.0188
Exact pair match      : 0.000
Parse success         : 1.000
```

State recall:

```text
Absent      : 0.000
New         : 0.000
Persistent  : 0.000
Resolved    : 0.750
```

---

## M7.3 QLoRA

```text
Accuracy              : 0.0286
Macro F1              : 0.0163
Exact pair match      : 0.000
Parse success         : 1.000
```

State recall:

```text
Absent      : 0.000
New         : 0.000
Persistent  : 0.000
Resolved    : 0.750
```

The adapter therefore produced essentially no downstream temporal improvement.

---

# 14. Raw Prediction Audit

Inspection of the raw generations showed that both the frozen and adapted model were producing nearly fixed temporal patterns.

Typical output resembled:

```text
atelectasis       → resolved
cardiomegaly      → new
consolidation     → resolved
edema             → resolved
pleural_effusion  → resolved
pneumonia         → resolved
pneumothorax      → resolved
```

This pattern was repeated across multiple patient pairs.

The behavior suggested that the model was learning prompt/output structure without adequately learning image-conditioned temporal discrimination.

---

# 15. First Diagnosis

The M7.3 results suggested three problems:

```text
1. loss was applied to the full conversation
2. the temporal data was strongly imbalanced
3. the evaluation split was too small
```

The next experiment therefore changed both the training objective and dataset construction.

---

# 16. Assistant-Only Loss

The original collator effectively used:

```python
labels = input_ids.clone()
```

with only padding masked.

This meant that the model was optimized on:

```text
system text
user text
image placeholders
assistant answer
```

The corrected objective instead masked the entire prompt:

```text
system tokens      → -100
user tokens        → -100
image prompt tokens→ -100
assistant tokens   → supervised labels
```

The intended optimization target became:

```text
assistant temporal JSON only
```

---

# 17. Why Assistant-Only Loss Matters

The actual task of interest is not:

> Reproduce the multimodal chat conversation.

It is:

> Produce the correct structured temporal interpretation.

Therefore, assistant-only loss concentrates the gradient on outputs such as:

```json
{
  "finding": "atelectasis",
  "change": "new"
}
```

instead of spending most of the learning signal on prompt reconstruction.

---

# 18. M7.5 Dataset Redesign

A substantially larger temporal SFT cohort was created.

Source dataset:

```text
Source pairs    : 61,219
Source patients : 12,051
```

Patient overlap:

```text
0
```

Selected splits:

| Split | Samples | Patients |
|---|---:|---:|
| Train | 3,000 | 1,975 |
| Validation | 400 | 260 |
| Test | 500 | 318 |

---

# 19. Patient-Disjoint Splitting

Patient splitting was performed before training pair sampling.

Conceptually:

```text
All patients
     │
     ├────────► train patients
     │
     ├────────► validation patients
     │
     └────────► test patients
```

Only after this split were training examples sampled.

This ensured that oversampling did not introduce patient leakage.

---

# 20. Change-Aware Pair Sampling

The training data was not sampled uniformly.

Pairs were categorized according to whether they contained:

```text
persistent
new
resolved
stable
```

Training pair targets were:

```text
persistent : 600
new        : 750
resolved   : 600
stable     : 1,050
```

Total:

```text
3,000 pairs
```

This corresponded approximately to:

```text
persistent : 20%
new        : 25%
resolved   : 20%
stable     : 35%
```

---

# 21. Why Pair-Level Balancing Was Chosen

Each training example contains seven finding labels.

Therefore, independently balancing every individual finding label is not straightforward.

The first correction instead balanced at the pair level:

```text
change-containing pairs
vs
stable pairs
```

This increased exposure to temporal changes without destroying the full seven-finding structure.

---

# 22. Remaining Finding-Level Imbalance

Despite pair-level rebalancing, the actual training labels remained strongly imbalanced.

Training state counts:

```text
absent      : 18,053
new         : 1,132
persistent  : 705
resolved    : 1,110
```

Total:

```text
21,000
```

because:

```text
3,000 pairs × 7 findings = 21,000 labels
```

Therefore:

```text
absent ≈ 86%
```

of the finding-level labels.

This became important in the final QLoRA result.

---

# 23. Validation Distribution

Validation state counts:

```text
absent      : 2,495
new         : 118
persistent  : 70
resolved    : 117
```

Validation pair categories:

```text
persistent : 66
new        : 80
resolved   : 67
stable     : 187
```

---

# 24. Test Distribution

The complete 500-pair held-out test cohort contained:

```text
absent      : 3,132
new         : 137
persistent  : 109
resolved    : 122
```

Pair categories:

```text
persistent : 96
new        : 89
resolved   : 62
stable     : 253
```

This was considerably more useful than the original 15-pair smoke test.

---

# 25. M7.5 Corrected QLoRA Configuration

The base model and LoRA architecture remained unchanged:

```text
Qwen2.5-VL-3B-Instruct

4-bit NF4

LoRA rank    : 8
LoRA alpha   : 16
LoRA dropout : 0.05
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
```

Training arguments included:

```text
batch size                 : 1
gradient accumulation      : 4
learning rate              : 2e-4
max steps                  : 20
fp16                       : True
gradient checkpointing     : True
optimizer                  : paged_adamw_8bit
```

---

# 26. M7.5 Training Behavior

The corrected assistant-only objective produced dramatically lower losses.

Representative training losses:

```text
Step 1  : 0.3160
Step 2  : 0.2866
Step 3  : 0.2096
Step 5  : 0.1359
Step 10 : 0.0706
Step 15 : 0.0808
Step 20 : 0.0400
```

The absolute values should not be compared directly with M7.3 because the supervised objective changed.

---

# 27. M7.5 Validation Loss

Validation loss decreased consistently:

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

At optimization level, the corrected training strategy therefore appeared highly successful.

---

# 28. Held-Out Evaluation

Before extending training to hundreds of optimizer steps, the adapter was tested on 100 held-out patient-disjoint pairs.

Results:

```text
Accuracy              : 0.8914
Macro F1              : 0.2356
Exact pair match      : 0.530
Parse success rate    : 1.000
```

At first glance, an accuracy of approximately 89% appeared promising.

The state-specific metrics showed otherwise.

---

# 29. State Recall

The M7.5 adapter produced:

```text
Absent recall      : 1.000
New recall         : 0.000
Persistent recall  : 0.000
Resolved recall    : 0.000
```

This means the model successfully learned the dominant class but failed to identify the non-absent temporal states.

---

# 30. Majority-State Collapse

The result represents a classic class-imbalance failure.

The model effectively learned:

```text
predict absent
```

for the large majority of finding-level outputs.

Since the dataset is strongly dominated by `absent`, this produces high aggregate accuracy.

However, the system fails on the actual longitudinal change states of interest.

---

# 31. Why 89% Accuracy Was Not Accepted

The adapted model achieved:

```text
Accuracy ≈ 89.1%
```

but:

```text
New recall        = 0%
Persistent recall = 0%
Resolved recall   = 0%
```

Therefore, the 89% accuracy cannot be interpreted as successful temporal adaptation.

The correct conclusion is:

> The adapter learned the majority temporal state rather than robust longitudinal change reasoning.

---

# 32. Macro F1 Revealed the Problem

Macro F1 was:

```text
0.2356
```

This value is substantially lower than the headline accuracy and reflects poor performance across the four classes.

The divergence between:

```text
Accuracy ≈ 0.89
```

and:

```text
Macro F1 ≈ 0.24
```

illustrates why class-sensitive evaluation is essential.

---

# 33. Why Training Was Stopped

The adapter had only completed 20 optimization steps.

It would have been possible to continue training for:

```text
100
200
750
```

or more optimizer steps.

However, the held-out evaluation already showed majority-class collapse.

Continuing the same objective would likely reinforce the dominant `absent` solution.

Therefore, further training was intentionally stopped.

---

# 34. Experimental Decision

The final decision was:

```text
Do not promote M7.5 QLoRA into default inference.
```

The v0.1.0 pipeline continues to use:

```text
base Qwen2.5-VL
        +
BiomedCLIP temporal models
        +
fusion
        +
safety-aware abstention
```

without a QLoRA adapter.

---

# 35. Why the Negative Result Is Kept

The QLoRA experiments are retained because they provide useful research evidence.

The experiments demonstrate that:

```text
successful fine-tuning mechanics
```

do not guarantee:

```text
successful downstream temporal reasoning
```

They also demonstrate that:

```text
low validation LM loss
```

does not guarantee:

```text
balanced temporal-state performance
```

This is important for biomedical multimodal fine-tuning.

---

# 36. Lessons from M7

## Lesson 1 — Training infrastructure is not model performance

The QLoRA pipeline worked technically.

The adapter loaded, trained, evaluated, and saved correctly.

That does not mean the adapted model solved the task.

---

## Lesson 2 — Loss must match the task

Training on the full conversation allowed the adapter to optimize prompt and formatting tokens.

Assistant-only masking produced a much cleaner temporal supervision objective.

---

## Lesson 3 — Pair balancing is not enough

Balancing change-containing pairs increased temporal examples.

However, each pair still contained multiple `absent` finding labels.

The finding-level distribution therefore remained heavily imbalanced.

---

## Lesson 4 — Accuracy is misleading under imbalance

The 89.1% result would appear successful if accuracy were reported alone.

State recall exposed the actual failure.

---

## Lesson 5 — Evaluate early

The adapter was evaluated after only 20 steps rather than after a full epoch.

This prevented unnecessary compute from being spent on a training setup already exhibiting majority-class collapse.

---

# 37. Recommended Next Experiment

The strongest next direction would be **per-finding temporal supervision**.

Instead of training one sample containing all seven finding states:

```text
Pair
│
├── atelectasis
├── cardiomegaly
├── consolidation
├── edema
├── pleural_effusion
├── pneumonia
└── pneumothorax
```

create independent training examples:

```text
Prior image
Current image
Target finding = consolidation
        ↓
Temporal state = new
```

This changes the supervision unit from:

```text
7 labels per example
```

to:

```text
1 controlled temporal label per example
```

---

# 38. Per-Finding Balanced QLoRA

A future dataset could explicitly target approximately balanced examples for:

```text
absent
new
persistent
resolved
```

within each finding where data availability permits.

Example:

```text
Finding: atelectasis

absent       N
new          N
persistent   N
resolved     N
```

This would remove the six-extra-absent-label problem created by seven-finding joint supervision.

---

# 39. Other Possible Improvements

Future experiments may investigate:

```text
class-weighted loss
focal-style objectives
hard-example mining
rare-state oversampling
per-finding adapters
longer training after balancing
larger LoRA rank
different LoRA targets
curriculum training
contrastive temporal objectives
```

These should be evaluated using patient-disjoint held-out cohorts.

---

# 40. Temporal Prompt Improvements

The prompt itself may also be improved.

Potential experiments include:

```text
single-finding questions
explicit prior/current definitions
state definitions in the prompt
few-shot temporal examples
structured constrained generation
```

However, prompt optimization should not replace proper dataset balancing.

---

# 41. Adapter Checkpoint Selection

Future experiments should evaluate multiple intermediate checkpoints.

For example:

```text
checkpoint 25
checkpoint 50
checkpoint 100
checkpoint 200
```

rather than automatically assuming the final checkpoint is optimal.

Task metrics should determine model selection.

---

# 42. Better Evaluation

Future QLoRA evaluation should include:

```text
full 500-pair held-out test set
per-finding confusion matrices
per-state recall
per-finding macro F1
balanced accuracy
exact pair match
format validity
calibration
```

The complete held-out test split is already substantially more useful than the original smoke cohort.

---

# 43. Comparison Framework

A future experiment should compare:

```text
Frozen Qwen2.5-VL
        vs
M7.3 naive QLoRA
        vs
M7.5 assistant-only QLoRA
        vs
balanced per-finding QLoRA
```

under the same:

```text
test patients
prompt structure
parser
metrics
generation settings
```

This would provide a clean ablation study.

---

# 44. Role of QLoRA in MedChange-VLM

QLoRA should be viewed as:

```text
experimental model adaptation research
```

rather than:

```text
required system dependency
```

The core MedChange architecture remains functional without it.

This separation ensures that an unsuccessful adaptation experiment does not destabilize the working safety-aware inference pipeline.

---

# 45. Relationship to the Main System

The current default pathway remains:

```text
Prior + Current X-ray
        │
        ├────────► BiomedCLIP temporal classifier
        │
        └────────► base Qwen2.5-VL reasoning
                         │
                         ▼
                    evidence fusion
                         │
                         ▼
                    safety policy
                         │
                         ▼
                    final report
```

The QLoRA branch exists alongside this pipeline as an experimental research track.

---

# 46. Reproducibility Artifacts

The repository retains:

```text
dataset construction scripts
training scripts
evaluation scripts
experiment summaries
dataset audit metadata
metric outputs
```

Generated datasets and adapter checkpoints are excluded from the final repository because they are reproducible artifacts and may contain machine-specific paths or large model files.

---

# 47. Generated Artifacts Excluded from Git

The following are intentionally not version-controlled:

```text
data/nih/qlora*/
models/qlora/
```

These include:

```text
train JSONL files
validation JSONL files
test JSONL files
adapter weights
trainer checkpoints
tokenizer snapshots
training state
```

The experiment code and summarized metrics remain version-controlled.

---

# 48. Current Status

The QLoRA research milestone can be summarized as:

```text
Dataset construction           ✓
Patient-disjoint splits        ✓
4-bit QLoRA setup              ✓
LoRA attachment                ✓
Multimodal training            ✓
Adapter saving                 ✓
Validation                     ✓
Assistant-only masking         ✓
Change-aware pair sampling     ✓
Held-out evaluation            ✓

Reliable change-state gain     ✗
Default-system integration     ✗
```

---

# 49. Final QLoRA Conclusion

The M7 experiments successfully established a reproducible QLoRA workflow for longitudinal multimodal adaptation of Qwen2.5-VL.

However, the experiments also demonstrated that:

```text
optimization success
```

and:

```text
structured output success
```

are not sufficient to establish:

```text
temporal reasoning success
```

The final assistant-only experiment achieved low validation loss and high aggregate accuracy, but failed to recall any evaluated `new`, `persistent`, or `resolved` states.

For this reason, QLoRA remains an experimental future-work branch rather than part of the MedChange-VLM v0.1.0 default inference architecture.

The next meaningful research step is not simply longer training.

It is a redesign of the supervision strategy to provide stronger, more balanced learning signals for non-absent temporal changes.