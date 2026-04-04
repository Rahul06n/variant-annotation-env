# 🧬 Variant Annotation Environment

> A real-world bioinformatics RL environment where AI agents classify genomic variants using the ACMG/AMP 5-tier framework — a task performed daily by clinical geneticists worldwide.

---

## 🌍 Motivation

Every year, millions of genetic variants are identified in patients undergoing diagnostic sequencing. Each variant must be classified into one of five ACMG/AMP tiers:

| Tier | Meaning |
|------|---------|
| Pathogenic | Causes disease |
| Likely Pathogenic | Probably causes disease |
| Uncertain Significance | Unknown impact |
| Likely Benign | Probably harmless |
| Benign | Harmless |

This classification is **slow, expert-intensive, and critical** — wrong classifications can lead to missed diagnoses or unnecessary interventions. This environment trains AI agents to perform variant classification across three levels of difficulty, using real-world evidence types drawn from clinical genetics practice.

---

## 🎮 Environment Overview

Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

**API:** Standard Gymnasium-style `reset()` / `step()` / `state()`

**Action space:** The agent submits:
- A classification (one of the 5 ACMG tiers)
- Evidence codes used (e.g. `["PVS1", "PM2", "PP1"]`)
- A reasoning string explaining the decision

**Observation space:** The agent receives:
- Variant identity (gene, HGVS notation, variant type)
- Task-specific evidence (structured data, clinical notes, or database entries)
- Feedback and reward after each step

---

## 📋 Tasks

### 🟢 Task 1 — Easy: Structured Evidence Classification

The agent receives fully pre-extracted, structured evidence:
- Variant type (frameshift, missense, etc.)
- Population frequency from gnomAD
- In-silico predictions (SIFT, PolyPhen2, CADD, REVEL)
- Functional study results
- Segregation data from family studies

**Expected difficulty:** A strong LLM with genomics knowledge should classify correctly most of the time.

---

### 🟡 Task 2 — Medium: Clinical Notes Classification

The agent receives **raw, unstructured clinical notes** as a clinician would write them. The agent must:
1. Extract relevant evidence from the notes
2. Map it to ACMG criteria
3. Reach a classification

**Expected difficulty:** Requires information extraction + clinical reasoning.

---

### 🔴 Task 3 — Hard: Conflicting Evidence Reconciliation

The agent receives **conflicting entries from multiple databases**:
- ClinVar submissions (multiple submitters, different classifications)
- gnomAD population frequency data
- Published literature with differing findings

The agent must reconcile the conflicts and reach a defensible final classification.

**Expected difficulty:** Even frontier models struggle with genuine conflicts.

---

## 🏆 Reward Function

Rewards provide **partial progress signals** throughout the episode — not just binary win/lose.

| Signal | Reward |
|--------|--------|
| Correct classification | +0.60 (easy) / +0.50 (medium) / +0.40 (hard) |
| Off-by-one tier | +0.30 (easy) / +0.25 (medium) / +0.20 (hard) |
| Correct evidence codes | Up to +0.25 |
| Reasoning provided | Up to +0.20 (easy) / +0.35 (hard) |
| Invalid classification | -0.20 |
| Extra steps beyond first | -0.10 per step |

Maximum reward per episode: **1.00**

---

## 📊 Baseline Scores

Evaluated using `gemini-2.0-flash` with 3 episodes per task:

| Task | Avg Reward | Min | Max |
|------|-----------|-----|-----|
| Easy | 0.450 | 0.150 | 0.750 |
| Medium | 0.267 | 0.100 | 0.600 |
| Hard | 0.200 | 0.200 | 0.200 |
| **Overall** | **0.306** | — | — |

The clear difficulty progression confirms the environment provides meaningful signal for RL training.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/variant_annotation_env
cd variant_annotation_env
pip install openenv-core
```

### Run the server locally

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Test with curl

```bash
# Reset environment
curl -X POST http://localhost:8000/reset

# Submit a classification
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "classification": "Pathogenic",
      "evidence_codes": ["PVS1", "PM2", "PP1"],
      "reasoning": "Frameshift variant causing premature stop codon. Absent from gnomAD. Segregates with disease.",
      "metadata": {}
    }
  }'
```

### Run baseline evaluation

```bash
export GEMINI_API_KEY="your-key-here"
python baseline.py
```

---

## 🗂️ Project Structure

```
variant_annotation_env/
├── models.py                          # Action, Observation Pydantic models
├── openenv.yaml                       # OpenEnv metadata
├── baseline.py                        # Baseline inference script
├── baseline_results.json              # Reproducible baseline scores
├── README.md                          # This file
└── server/
    ├── app.py                         # FastAPI application
    ├── variant_annotation_env_environment.py  # Environment logic + graders
    ├── Dockerfile                     # Container definition
    └── requirements.txt               # Dependencies
```

---

## 🧬 Variant Data

The environment includes real-world inspired variants from BRCA1 and BRCA2 genes, covering all 5 classification tiers:

- **Pathogenic:** Frameshift variants with loss of function (e.g. BRCA1 c.5266dupC)
- **Likely Pathogenic:** Novel truncating variants with limited evidence
- **Uncertain Significance:** Missense variants with conflicting in-silico predictions
- **Likely Benign:** Common missense variants with normal functional assays
- **Benign:** Well-established polymorphisms present in >20% of population

---

## 📖 ACMG/AMP Classification Framework

This environment implements the [ACMG/AMP 2015 variant classification guidelines](https://www.acmg.net/docs/standards_guidelines_for_the_interpretation_of_sequence_variants.pdf), the international standard used by clinical laboratories worldwide.

---

## 👤 Author

Built for the Meta PyTorch OpenEnv Hackathon x SST 2026.
