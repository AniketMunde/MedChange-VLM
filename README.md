# MedChange-VLM

### Safety-Aware Multimodal AI for Longitudinal Chest X-Ray Change Analysis

MedChange-VLM is a research prototype for **longitudinal chest X-ray analysis** that compares a prior and current radiograph to identify clinically relevant temporal changes.

The system combines a **BiomedCLIP-based temporal classifier** with **Qwen2.5-VL multimodal reasoning**, reconciles their predictions through evidence-aware fusion, estimates uncertainty, and selectively abstains when model disagreement makes an automated conclusion unreliable.

Rather than optimizing only for prediction coverage, MedChange-VLM is designed around a central principle:

> **When multimodal evidence is insufficient or conflicting, the system should explicitly express uncertainty and route the finding for review rather than silently forcing a prediction.**

The current release supports seven chest X-ray findings and provides a complete research workflow including patient-aware longitudinal dataset construction, temporal modeling, multimodal reasoning, selective-risk evaluation, evidence-grounded reporting, a FastAPI inference service, and a Streamlit dashboard.

> **Research prototype only. MedChange-VLM is not a medical device and must not be used for clinical diagnosis or patient-care decisions.**

---

## Overview

Most medical-image models analyze a single study independently.

In clinical practice, however, interpretation frequently depends on **change over time**:

- Has an abnormality newly appeared?
- Is a previous finding still present?
- Has it resolved?
- Is there no evidence of the finding in either study?

MedChange-VLM formulates this as a longitudinal multimodal reasoning problem.

For every supported finding, the system predicts one of four temporal states:

| Temporal State | Interpretation |
|---|---|
| `absent` | Finding is absent in both studies |
| `new` | Finding appears in the current study |
| `persistent` | Finding is present in both studies |
| `resolved` | Finding was present previously but is absent currently |

The system then combines independent evidence from a vision-language model and a learned temporal classifier before deciding whether to return a prediction or request review.

---

## Supported Findings

MedChange-VLM v0.1.0 is intentionally restricted to seven target findings:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Pleural effusion
6. Pneumonia
7. Pneumothorax

Predictions outside this validated research scope should not be interpreted as supported system behavior.

---

# System Architecture

```text
                    Prior Chest X-Ray
                            +
                   Current Chest X-Ray
                            │
                            ▼
                  ┌──────────────────┐
                  │ Input Validation │
                  │ + Safety Guards  │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌─────────────────┐       ┌──────────────────┐
     │   BiomedCLIP    │       │   Qwen2.5-VL    │
     │                 │       │      3B          │
     │ Prior / Current │       │ Multimodal       │
     │ Embeddings      │       │ Temporal         │
     │       +         │       │ Reasoning        │
     │ Temporal Models│       │                  │
     └────────┬────────┘       └────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Evidence Fusion │
                  └────────┬────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Agreement /       │
                 │ Conflict Analysis │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Selective-Risk    │
                 │ Safety Policy     │
                 └─────────┬─────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
       Accepted Prediction         Abstention
                                       │
                                       ▼
                                  Human Review
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Evidence-Grounded │
                 │ Longitudinal      │
                 │ Report            │
                 └─────────┬─────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               FastAPI          Streamlit
```

---

# Core Components

## 1. Longitudinal Dataset Construction

MedChange-VLM constructs longitudinal study pairs from the NIH ChestX-ray14 dataset.

The pipeline includes:

- patient-level study grouping
- chronological study ordering
- prior/current pair construction
- same-view filtering
- temporal state derivation
- patient-aware train/test splitting
- reproducible seeded evaluation
- leakage auditing

Patient-level separation is enforced during evaluation so that the same patient does not appear in both training and held-out evaluation cohorts.

---

## 2. BiomedCLIP Temporal Modeling

The vision branch uses:

**BiomedCLIP-PubMedBERT_256-vit_base_patch16_224**

For every longitudinal pair, MedChange-VLM extracts representations from:

```text
Prior image
Current image
Delta representation
```

Finding-specific temporal classifiers can use combinations of:

```text
prior
current
prior + current
prior + current + delta
```

depending on the selected feature configuration.

The classifiers predict:

```text
absent
new
persistent
resolved
```

for each target finding.

---

## 3. Qwen2.5-VL Temporal Reasoning

The second reasoning pathway uses:

**Qwen/Qwen2.5-VL-3B-Instruct**

with 4-bit quantization for resource-efficient local inference.

The model receives the prior and current radiographs together and performs structured temporal reasoning across the supported findings.

Outputs are parsed into a controlled representation containing temporal states and evidence used by the fusion layer.

Robust parsing and validation prevent malformed model responses from silently entering downstream reasoning.

---

## 4. Multimodal Evidence Fusion

MedChange-VLM does not treat either model as an unquestioned authority.

Instead, predictions from BiomedCLIP and Qwen are compared finding by finding.

The fusion layer considers:

- BiomedCLIP temporal state
- BiomedCLIP confidence
- Qwen temporal state
- Qwen confidence
- model agreement
- model conflict
- uncertainty
- configured safety policy

This produces a final temporal state together with a decision reason and review flag.

---

# Safety-Aware Selective Prediction

A major design goal of MedChange-VLM is **selective prediction**.

The system can abstain instead of forcing a prediction when evidence is unreliable.

Supported safety-policy experiments include:

```text
strict
change_sensitive
confidence_margin
low_confidence_only
```

with configurable confidence thresholds.

The default research workflow uses a change-sensitive policy.

For example:

```text
BiomedCLIP : absent
Qwen       : new
Finding    : atelectasis

Agreement  : conflict
Uncertainty: high
Final      : uncertain
Review     : required
```

Instead of hiding the disagreement, MedChange-VLM exposes it directly.

---

# Evidence-Grounded Reporting

The final report is generated from the fused structured result rather than allowing an unconstrained language model to invent the final conclusion.

A representative output is:

```text
Overall change : uncertain
Uncertainty    : high
Review needed  : True

IMPRESSION

Uncertain due to model disagreement or insufficient
agreement: atelectasis.

UNCERTAIN / CONFLICTING FINDINGS

- atelectasis: The current image shows a more prominent
  opacity in the left lung field compared to the prior image.

REVIEW FLAGS

- atelectasis:
  BiomedCLIP=absent
  Qwen=new
  agreement=conflict
  uncertainty=high
```

This makes the reasoning path inspectable and keeps model disagreement visible to the user.

---

# Evaluation

## Patient-Aware Evaluation

The final fusion benchmark was evaluated across multiple patient-disjoint splits.

Five random seeds were used:

```text
11
21
42
84
123
```

Patient overlap between train and test cohorts:

```text
0
```

### Finding-Level Results

| Model | Coverage | Selective Accuracy | Macro F1 | Covered Error |
|---|---:|---:|---:|---:|
| BiomedCLIP | 1.000 ± 0.000 | 0.757 ± 0.023 | 0.354 ± 0.032 | 0.243 ± 0.023 |
| Qwen2.5-VL | 1.000 ± 0.000 | 0.694 ± 0.046 | 0.269 ± 0.023 | 0.306 ± 0.046 |
| **MedChange** | **0.709 ± 0.038** | **0.865 ± 0.012** | 0.275 ± 0.043 | **0.135 ± 0.012** |

MedChange deliberately sacrifices coverage in exchange for a lower covered error rate.

Approximately:

```text
Coverage    : 70.9%
Review rate : 29.1%
Accuracy on covered predictions : 86.5%
Covered error rate              : 13.5%
```

This trade-off is intentional.

The system is designed to route uncertain cases to review rather than maximize automated coverage.

---

## State Recall

### BiomedCLIP

```text
Absent      : 0.902 ± 0.025
New         : 0.172 ± 0.087
Persistent  : 0.117 ± 0.048
Resolved    : 0.198 ± 0.089
```

### Qwen2.5-VL

```text
Absent      : 0.834 ± 0.049
New         : 0.295 ± 0.122
Persistent  : 0.000 ± 0.000
Resolved    : 0.046 ± 0.054
```

### MedChange

```text
Absent      : 0.988 ± 0.011
New         : 0.095 ± 0.104
Persistent  : 0.000 ± 0.000
Resolved    : 0.025 ± 0.056
```

These results also expose an important limitation: non-absent temporal states remain substantially harder than identifying stable absence.

This limitation motivates future work on balanced temporal supervision and improved multimodal temporal representation learning.

---

# Selective-Risk Analysis

Multiple abstention strategies were evaluated across confidence thresholds.

For example, the `strict` policy at threshold `0.80` produced approximately:

```text
Coverage           : 0.669
Selective accuracy : 0.882
Covered error      : 0.118
```

while a `confidence_margin` policy at `0.80` produced:

```text
Coverage           : 0.866
Selective accuracy : 0.805
Covered error      : 0.195
```

These experiments demonstrate the expected risk-coverage trade-off:

> Increasing abstention can reduce the error rate among predictions that the system chooses to return.

The operating point should therefore be selected according to the intended research setting rather than accuracy alone.

---

# Experimental QLoRA Adaptation

Parameter-efficient temporal adaptation of Qwen2.5-VL was also investigated.

This work is **experimental and is not used by the default MedChange-VLM inference pipeline**.

The adaptation pipeline includes:

- 4-bit model quantization
- QLoRA
- LoRA rank 8
- LoRA alpha 16
- dropout 0.05
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- patient-disjoint SFT construction
- assistant-only loss masking
- temporal change-aware sampling

A larger temporal SFT cohort was constructed from:

```text
Source longitudinal pairs : 61,219
Source patients           : 12,051
Patient overlap           : 0
```

The final experimental split contained:

| Split | Pairs |
|---|---:|
| Train | 3,000 |
| Validation | 400 |
| Test | 500 |

The training cohort intentionally increased representation of:

```text
new
persistent
resolved
```

pairs.

### Assistant-Only QLoRA Smoke Experiment

A 20-step QLoRA experiment reduced validation language-model loss to:

```text
0.0597
```

However, held-out temporal evaluation revealed class collapse:

```text
Accuracy       : 0.8914
Macro F1       : 0.2356

Absent recall      : 1.000
New recall         : 0.000
Persistent recall  : 0.000
Resolved recall    : 0.000
```

The apparently high accuracy was therefore driven by the dominant `absent` state and **does not represent improved temporal reasoning**.

For this reason, the QLoRA adapter is not enabled in the default inference system.

This negative result is retained as part of the research workflow because it demonstrates why aggregate accuracy alone is insufficient for highly imbalanced longitudinal medical tasks.

---

# API

MedChange-VLM exposes its inference workflow through FastAPI.

Start the API with:

```bash
uvicorn medchange.api.app:app --app-dir src --reload
```

The API provides endpoints including:

```text
GET  /health
GET  /model-info
GET  /runtime-status
POST /analyze-pair
POST /cache/clear
```

The `/analyze-pair` endpoint accepts:

```text
prior image
current image
pair ID
prior study ID
current study ID
safety policy
safety threshold
```

and returns structured longitudinal findings, uncertainty, review flags, report impression, safety configuration, and runtime information.

---

## API Safety

The API includes:

- uploaded-content validation
- image-format checks
- longitudinal-pair validation
- identical-image rejection
- safety-policy validation
- request-data validation
- runtime busy handling
- structured API errors
- inference exception handling
- request caching
- runtime status monitoring

A busy inference runtime can return a retryable `503` response rather than launching competing GPU workloads.

---

# Streamlit Dashboard

A Streamlit interface provides an interactive research demonstration of the complete MedChange workflow.

Start the dashboard with:

```bash
streamlit run src/medchange/dashboard/app.py
```

The dashboard provides:

- prior/current X-ray upload
- API health status
- longitudinal input validation
- study metadata
- configurable safety policy
- finding-level temporal predictions
- model agreement visualization
- uncertainty indicators
- human-review flags
- evidence-grounded impression
- runtime/cache status
- explicit validated-scope warning

The interface intentionally displays disagreement rather than hiding it behind a single model score.

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd Medchange-VLM
```

## 2. Create a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

GPU-enabled PyTorch should be installed according to the CUDA version available on the target system.

---

# Running the Tests

Run the complete test suite:

```bash
python -m pytest -v
```

The tests cover major components including:

- medical schemas
- NIH dataset adapters
- streaming data
- longitudinal pair construction
- temporal models
- model agreement
- Qwen output parsing
- safety policies
- API validation
- API busy-state behavior
- runtime management
- cache behavior

---

# Project Structure

```text
Medchange-VLM/
│
├── src/
│   └── medchange/
│       ├── api/
│       ├── dashboard/
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       ├── reasoning/
│       ├── runtime/
│       ├── safety/
│       └── training/
│
├── scripts/
│   ├── build_longitudinal_pairs.py
│   ├── build_temporal_sft_dataset.py
│   ├── build_m75_qlora_dataset.py
│   ├── train_temporal_qlora.py
│   ├── evaluate_temporal_qlora.py
│   └── ...
│
├── tests/
│
├── experiments/
│   ├── temporal_ablation/
│   ├── medchange_m55_oof/
│   ├── medchange_m552/
│   ├── qlora_m74/
│   └── qlora_m75/
│
├── models/
│   └── temporal/
│
├── docs/
│   └── images/
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

Generated datasets, downloaded foundation-model weights, caches, and QLoRA checkpoints are intentionally excluded from version control.

---

# Reproducibility

The evaluation pipeline uses:

- explicit random seeds
- patient-aware splitting
- patient-overlap auditing
- deterministic dataset construction where applicable
- stored experiment summaries
- finding-level metrics
- state-specific recall
- selective-risk metrics
- coverage/error analysis

Patient-level separation is especially important because random image-level splitting can produce overly optimistic biomedical evaluation when multiple studies from the same patient appear across splits.

---

# Design Principles

MedChange-VLM was developed around several principles.

### Longitudinal reasoning over isolated classification

The task is explicitly formulated as change detection between prior and current studies.

### Independent evidence pathways

BiomedCLIP and Qwen provide complementary evidence rather than relying on a single model.

### Uncertainty is a result

`uncertain` is treated as meaningful system behavior rather than an implementation failure.

### Selective prediction

The model is allowed to abstain.

### Human review

Conflicting or insufficient evidence can trigger explicit review.

### Patient-aware evaluation

Evaluation prevents patient leakage between training and held-out cohorts.

### Transparent negative results

Experiments such as QLoRA adaptation are reported even when they do not improve temporal-state performance.

---

# Limitations

MedChange-VLM v0.1.0 has several important limitations.

### Limited finding scope

The current system supports only seven predefined chest X-ray findings.

### Dataset limitations

Development and evaluation are based primarily on NIH ChestX-ray14-derived longitudinal data.

Performance may not transfer to other institutions, acquisition systems, patient populations, or clinical settings.

### Label quality

Temporal states are derived from available dataset labels rather than expert-authored longitudinal comparison reports.

### Class imbalance

`absent` is substantially more common than `new`, `persistent`, and `resolved`.

This particularly affects temporal-state recall.

### VLM reasoning limitations

Qwen2.5-VL is a general-purpose multimodal foundation model and is not a clinically validated radiology model.

### No clinical validation

The system has not undergone prospective clinical evaluation or regulatory validation.

### No diagnostic claim

Outputs must not be interpreted as medical diagnoses.

---

# Future Work

The next research directions include:

### 1. State-Balanced Temporal Adaptation

Develop per-finding or otherwise state-balanced supervision for:

```text
absent
new
persistent
resolved
```

to address the class-collapse observed during QLoRA experiments.

### 2. Lesion and Anatomical Grounding

Connect temporal predictions to spatial evidence through:

- bounding boxes
- anatomical regions
- attention/heat-map analysis
- grounding-aware VLM reasoning

### 3. Agentic Verification

Introduce explicit verification and adjudication components capable of:

```text
initial analysis
      ↓
evidence verification
      ↓
conflict detection
      ↓
targeted re-analysis
      ↓
adjudication
      ↓
human escalation
```

The goal is not to replace deterministic safety policies, but to provide additional evidence checking before final escalation.

### 4. External Validation

Evaluate on independent chest X-ray datasets and institutionally distinct cohorts.

### 5. Better Calibration

Improve uncertainty calibration and evaluate risk-coverage behavior on larger held-out cohorts.

### 6. Expert Evaluation

Compare system-generated temporal interpretations with radiologist longitudinal assessments.

### 7. DICOM and Clinical Workflow Support

Future engineering work may extend the research pipeline toward DICOM ingestion and integration with medical-imaging research workflows.

---

# Research Roadmap

```text
Longitudinal Pairing
        ✓
        │
Temporal BiomedCLIP Modeling
        ✓
        │
Qwen2.5-VL Temporal Reasoning
        ✓
        │
Evidence Fusion
        ✓
        │
Verification + Uncertainty
        ✓
        │
Selective-Risk Safety
        ✓
        │
FastAPI + Streamlit
        ✓
        │
Patient-Aware Evaluation
        ✓
        │
QLoRA Temporal Adaptation
        ◐ Experimental
        │
Balanced Temporal QLoRA
        ○ Future
        │
Lesion Grounding
        ○ Future
        │
Agentic Verification
        ○ Future
        │
External / Expert Validation
        ○ Future
```

---

# Technology Stack

Core technologies include:

```text
Python
PyTorch
Transformers
PEFT / QLoRA
bitsandbytes
BiomedCLIP
Qwen2.5-VL
scikit-learn
Pandas
NumPy
Pillow
FastAPI
Pydantic
Streamlit
pytest
```

---

# Research Disclaimer

**MedChange-VLM is an experimental research prototype.**

It is not approved as a medical device and has not been clinically validated.

The software and generated outputs must not be used to:

- diagnose disease
- determine treatment
- replace radiologist interpretation
- make clinical decisions
- provide patient-specific medical advice

All results should be interpreted as research outputs only.

---

# Citation

If you use MedChange-VLM in research or build upon the project, please cite the repository.

A formal `CITATION.cff` will be provided with the research release.

---

# License

See the repository `LICENSE` file for licensing information.

---

# Acknowledgements

This project builds upon open-source research and tooling from the biomedical imaging, vision-language modeling, and machine-learning communities, including BiomedCLIP, Qwen2.5-VL, Hugging Face Transformers, PyTorch, and the NIH ChestX-ray14 dataset ecosystem.

---

## MedChange-VLM v0.1.0

**Longitudinal reasoning. Independent multimodal evidence. Explicit uncertainty. Selective prediction. Human review.**