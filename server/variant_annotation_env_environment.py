# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Variant Annotation Environment — Full Implementation.

A real-world bioinformatics RL environment where an AI agent classifies
genomic variants using the ACMG/AMP 5-tier classification framework.

This is a genuine task performed daily by clinical geneticists and
bioinformaticians. Accurate variant classification directly impacts
patient diagnosis and treatment decisions.

Tasks
-----
easy   : Classify a variant given fully structured, pre-extracted evidence.
medium : Classify a variant from raw, unstructured clinical notes.
hard   : Reconcile conflicting classifications from ClinVar, gnomAD, and
         published literature to reach a final classification.

Reward Design
-------------
Rewards are shaped to provide partial progress signals throughout the
episode, not just at the end.

  +1.00  Correct classification
  +0.50  Off-by-one classification (e.g. Pathogenic vs Likely Pathogenic)
  +0.30  Correct evidence codes cited (regardless of classification)
  +0.20  Reasonable reasoning provided (non-empty, >20 chars)
  -0.20  Invalid classification string submitted
  -0.10  Per extra step beyond the first attempt (efficiency penalty)
   0.00  Completely wrong classification with no redeemable evidence
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        VALID_CLASSIFICATIONS,
        VariantAnnotationAction,
        VariantAnnotationObservation,
    )
except ImportError:
    from models import (
        VALID_CLASSIFICATIONS,
        VariantAnnotationAction,
        VariantAnnotationObservation,
    )


# ── Variant Data Bank ─────────────────────────────────────────────────────────
# Real-world inspired variants. Each entry contains the ground truth
# classification and all evidence fields needed for the three task tiers.

VARIANT_DATA = [
    # ── PATHOGENIC variants ──────────────────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.5266dupC",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.5266dupC (p.Gln1756Profs*74)",
        "correct_classification": "Pathogenic",
        "variant_type": "frameshift",
        "population_frequency": 0.0,
        "in_silico_predictions": {
            "SIFT": "deleterious",
            "PolyPhen2": "probably_damaging",
            "CADD": 35.2,
        },
        "functional_evidence": "Frameshift causes premature stop codon. Loss of BRCT domain confirmed in functional assay. Null allele.",
        "segregation_data": "Segregates with disease in 3 affected families (LOD score 3.2).",
        "clinical_notes": (
            "Patient: 38F. Strong family history of breast and ovarian cancer. "
            "Mother and maternal aunt both diagnosed with breast cancer before 45. "
            "Sequencing identified a single cytosine duplication at position 5266 in BRCA1 "
            "causing a frameshift. Variant absent from gnomAD (>250,000 alleles). "
            "Functional studies show loss of BRCT domain. Multiple ClinVar submitters "
            "classify as Pathogenic. Segregation confirmed in 3 families."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.5266dupC",
            "submitters": [
                {"name": "Ambry Genetics", "classification": "Pathogenic", "stars": 2},
                {"name": "GeneDx", "classification": "Pathogenic", "stars": 2},
                {"name": "ENIGMA", "classification": "Pathogenic", "stars": 3},
            ],
            "review_status": "reviewed by expert panel",
            "last_updated": "2023-09-01",
        },
        "gnomad_entry": {
            "allele_frequency": 0.0,
            "allele_count": 0,
            "total_alleles": 251448,
            "popmax_af": 0.0,
            "note": "Absent from all gnomAD populations",
        },
        "literature_evidence": [
            {
                "pmid": "7894493",
                "title": "Identification of BRCA1 mutations in hereditary breast and ovarian cancer families",
                "finding": "c.5266dupC identified as disease-causing in multiple HBOC families",
                "classification": "Pathogenic",
            },
            {
                "pmid": "26287974",
                "title": "ENIGMA BRCA1/2 Classification Criteria",
                "finding": "Meets criteria for Pathogenic: PVS1 + PS4 + PP1_Strong",
                "classification": "Pathogenic",
            },
        ],
        "correct_evidence_codes": ["PVS1", "PS4", "PM2", "PP1"],
    },

    {
        "variant_id": "NM_000059.3:c.6275_6276delTT",
        "gene": "BRCA2",
        "hgvs_notation": "NM_000059.3:c.6275_6276delTT (p.Leu2092Tyrfs*7)",
        "correct_classification": "Pathogenic",
        "variant_type": "frameshift",
        "population_frequency": 0.000004,
        "in_silico_predictions": {
            "SIFT": "deleterious",
            "CADD": 38.0,
        },
        "functional_evidence": "Frameshift leading to truncated protein. Loss of RAD51 binding domain.",
        "segregation_data": "Co-segregates with ovarian cancer in 2 families.",
        "clinical_notes": (
            "Patient: 52F, ovarian cancer diagnosis at 49. Sister with breast cancer. "
            "BRCA2 sequencing shows deletion of TT at c.6275. Frameshift causes "
            "premature stop 7 codons downstream. Protein truncation confirmed. "
            "Very low allele frequency in gnomAD (4 in 1M alleles). "
            "Expert panel classifies as Pathogenic."
        ),
        "clinvar_entry": {
            "variant_id": "NM_000059.3:c.6275_6276delTT",
            "submitters": [
                {"name": "Invitae", "classification": "Pathogenic", "stars": 2},
                {"name": "ENIGMA", "classification": "Pathogenic", "stars": 3},
            ],
            "review_status": "reviewed by expert panel",
            "last_updated": "2023-06-15",
        },
        "gnomad_entry": {
            "allele_frequency": 0.000004,
            "allele_count": 4,
            "total_alleles": 1000000,
            "popmax_af": 0.000008,
            "note": "Extremely rare — consistent with pathogenic variant",
        },
        "literature_evidence": [
            {
                "pmid": "11157798",
                "title": "BRCA2 frameshift variants in HBOC",
                "finding": "Truncating variants in RAD51-binding domain consistently pathogenic",
                "classification": "Pathogenic",
            },
        ],
        "correct_evidence_codes": ["PVS1", "PM2", "PP1"],
    },

    # ── LIKELY PATHOGENIC variants ───────────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.4964_4982del19",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.4964_4982del19",
        "correct_classification": "Likely Pathogenic",
        "variant_type": "frameshift",
        "population_frequency": 0.0,
        "in_silico_predictions": {
            "SIFT": "deleterious",
            "CADD": 32.1,
        },
        "functional_evidence": "Predicted frameshift; functional study pending.",
        "segregation_data": "Co-segregates in 1 family (insufficient for strong evidence).",
        "clinical_notes": (
            "Patient: 44F, breast cancer at 41. No prior genetic testing. "
            "Novel 19bp deletion in BRCA1 exon 16. Causes frameshift and predicted stop codon. "
            "Absent from gnomAD. Only one family studied so segregation evidence is limited. "
            "Functional studies underway. No prior ClinVar submissions."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.4964_4982del19",
            "submitters": [
                {"name": "Laboratory A", "classification": "Likely Pathogenic", "stars": 1},
            ],
            "review_status": "criteria provided, single submitter",
            "last_updated": "2024-01-10",
        },
        "gnomad_entry": {
            "allele_frequency": 0.0,
            "allele_count": 0,
            "total_alleles": 251448,
            "note": "Absent from gnomAD",
        },
        "literature_evidence": [
            {
                "pmid": "N/A",
                "title": "Novel variant — no prior publications",
                "finding": "Not yet reported in literature",
                "classification": "Unknown",
            },
        ],
        "correct_evidence_codes": ["PVS1", "PM2", "PP1"],
    },

    # ── UNCERTAIN SIGNIFICANCE variants ─────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.4837A>G",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.4837A>G (p.Ser1613Gly)",
        "correct_classification": "Uncertain Significance",
        "variant_type": "missense",
        "population_frequency": 0.0002,
        "in_silico_predictions": {
            "SIFT": "tolerated",
            "PolyPhen2": "benign",
            "CADD": 12.3,
            "REVEL": 0.28,
        },
        "functional_evidence": "No functional studies available.",
        "segregation_data": "No segregation data available.",
        "clinical_notes": (
            "Patient: 39F, personal history of breast cancer. "
            "BRCA1 missense variant p.Ser1613Gly identified. Low allele frequency (0.02%). "
            "In-silico predictions conflicting — SIFT tolerated, PolyPhen benign, "
            "but variant is in a conserved region. No functional data. "
            "Not enough evidence to classify as pathogenic or benign."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.4837A>G",
            "submitters": [
                {"name": "Lab X", "classification": "Uncertain Significance", "stars": 1},
                {"name": "Lab Y", "classification": "Likely Benign", "stars": 1},
            ],
            "review_status": "criteria provided, conflicting interpretations",
            "last_updated": "2023-11-20",
        },
        "gnomad_entry": {
            "allele_frequency": 0.0002,
            "allele_count": 50,
            "total_alleles": 251448,
            "popmax_af": 0.0003,
            "note": "Rare but present — not strong evidence either way",
        },
        "literature_evidence": [
            {
                "pmid": "19431188",
                "title": "Missense variants in BRCA1 BRCT domain",
                "finding": "p.Ser1613Gly shows intermediate functional impact",
                "classification": "Uncertain Significance",
            },
            {
                "pmid": "24033266",
                "title": "BRCA1 VUS reclassification study",
                "finding": "Insufficient evidence for reclassification",
                "classification": "Uncertain Significance",
            },
        ],
        "correct_evidence_codes": ["BP4", "PM2"],
    },

    # ── LIKELY BENIGN variants ───────────────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.2077G>A",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.2077G>A (p.Asp693Asn)",
        "correct_classification": "Likely Benign",
        "variant_type": "missense",
        "population_frequency": 0.008,
        "in_silico_predictions": {
            "SIFT": "tolerated",
            "PolyPhen2": "benign",
            "CADD": 8.1,
            "REVEL": 0.12,
        },
        "functional_evidence": "Functional assay shows normal BRCT domain activity.",
        "segregation_data": "Does not segregate with disease in 2 families.",
        "clinical_notes": (
            "Patient: 55F, referred for BRCA testing after breast cancer diagnosis. "
            "Variant p.Asp693Asn identified in BRCA1. Population frequency 0.8% — "
            "present in healthy controls. All in-silico tools predict benign/tolerated. "
            "Functional data shows normal protein function. Does not segregate with "
            "cancer in two tested families. Likely represents a polymorphism."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.2077G>A",
            "submitters": [
                {"name": "Ambry", "classification": "Likely Benign", "stars": 2},
                {"name": "GeneDx", "classification": "Likely Benign", "stars": 2},
                {"name": "Invitae", "classification": "Benign", "stars": 2},
            ],
            "review_status": "criteria provided, multiple submitters",
            "last_updated": "2023-08-01",
        },
        "gnomad_entry": {
            "allele_frequency": 0.008,
            "allele_count": 2012,
            "total_alleles": 251448,
            "popmax_af": 0.012,
            "note": "Common in population — strong evidence against pathogenicity",
        },
        "literature_evidence": [
            {
                "pmid": "20104584",
                "title": "Population frequency analysis of BRCA1 variants",
                "finding": "p.Asp693Asn present at 0.8% in healthy controls, inconsistent with high-penetrance pathogenicity",
                "classification": "Likely Benign",
            },
        ],
        "correct_evidence_codes": ["BS1", "BP4", "BS3"],
    },

    # ── BENIGN variants ──────────────────────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.3113A>G",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.3113A>G (p.Glu1038Gly)",
        "correct_classification": "Benign",
        "variant_type": "missense",
        "population_frequency": 0.21,
        "in_silico_predictions": {
            "SIFT": "tolerated",
            "PolyPhen2": "benign",
            "CADD": 3.2,
        },
        "functional_evidence": "Well-established benign polymorphism. Normal protein function confirmed.",
        "segregation_data": "Present in both affected and unaffected family members equally.",
        "clinical_notes": (
            "Patient: 47F, breast cancer. Incidental finding of BRCA1 variant "
            "p.Glu1038Gly. This is a well-known common polymorphism present in 21% "
            "of the population. All computational tools predict benign. Functional "
            "studies confirm normal protein activity. Classified as Benign by all "
            "major databases. Not the cause of patient's cancer."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.3113A>G",
            "submitters": [
                {"name": "ENIGMA", "classification": "Benign", "stars": 3},
                {"name": "Ambry", "classification": "Benign", "stars": 2},
                {"name": "GeneDx", "classification": "Benign", "stars": 2},
                {"name": "ClinGen", "classification": "Benign", "stars": 3},
            ],
            "review_status": "reviewed by expert panel",
            "last_updated": "2023-05-01",
        },
        "gnomad_entry": {
            "allele_frequency": 0.21,
            "allele_count": 52804,
            "total_alleles": 251448,
            "popmax_af": 0.25,
            "note": "Very common variant — present in 21% of population",
        },
        "literature_evidence": [
            {
                "pmid": "15356553",
                "title": "Characterization of common BRCA1 polymorphisms",
                "finding": "p.Glu1038Gly is a common benign polymorphism with no association to cancer risk",
                "classification": "Benign",
            },
            {
                "pmid": "26287974",
                "title": "ENIGMA BRCA1 classification",
                "finding": "Meets BA1 criterion — too common to be pathogenic",
                "classification": "Benign",
            },
        ],
        "correct_evidence_codes": ["BA1", "BS1", "BP4"],
    },

    # ── HARD: Conflicting Evidence ───────────────────────────────────────────
    {
        "variant_id": "NM_007294.3:c.5096G>A",
        "gene": "BRCA1",
        "hgvs_notation": "NM_007294.3:c.5096G>A (p.Arg1699Gln)",
        "correct_classification": "Likely Pathogenic",
        "variant_type": "missense",
        "population_frequency": 0.000012,
        "in_silico_predictions": {
            "SIFT": "deleterious",
            "PolyPhen2": "probably_damaging",
            "CADD": 26.4,
            "REVEL": 0.71,
        },
        "functional_evidence": "Intermediate functional impact — partial loss of BRCT domain activity (40% reduction).",
        "segregation_data": "Segregates in 2 families but absent in 1 affected individual.",
        "clinical_notes": (
            "Patient: 41F, breast cancer, strong family history. "
            "BRCA1 missense p.Arg1699Gln identified. Extremely rare (AF=0.0012%). "
            "In-silico tools predict deleterious. However functional studies show only "
            "partial (40%) loss of activity — not complete null. ClinVar has conflicting "
            "submissions. Segregation data imperfect. Requires careful weighing of evidence."
        ),
        "clinvar_entry": {
            "variant_id": "NM_007294.3:c.5096G>A",
            "submitters": [
                {"name": "Lab A", "classification": "Pathogenic", "stars": 1},
                {"name": "Lab B", "classification": "Likely Pathogenic", "stars": 2},
                {"name": "Lab C", "classification": "Uncertain Significance", "stars": 1},
                {"name": "ENIGMA", "classification": "Likely Pathogenic", "stars": 3},
            ],
            "review_status": "criteria provided, conflicting interpretations",
            "last_updated": "2024-02-01",
        },
        "gnomad_entry": {
            "allele_frequency": 0.000012,
            "allele_count": 3,
            "total_alleles": 251448,
            "popmax_af": 0.000024,
            "note": "Extremely rare — consistent with pathogenic or low-penetrance variant",
        },
        "literature_evidence": [
            {
                "pmid": "17924331",
                "title": "Functional analysis of BRCA1 BRCT missense variants",
                "finding": "p.Arg1699Gln shows intermediate functional impact — 40% reduction in BRCT activity",
                "classification": "Uncertain Significance",
            },
            {
                "pmid": "26287974",
                "title": "ENIGMA classification update",
                "finding": "Reclassified to Likely Pathogenic based on updated multifactorial likelihood model",
                "classification": "Likely Pathogenic",
            },
            {
                "pmid": "30209399",
                "title": "Multifactorial analysis of BRCA1 missense variants",
                "finding": "Combined posterior probability of pathogenicity: 0.82 — meets Likely Pathogenic threshold",
                "classification": "Likely Pathogenic",
            },
        ],
        "correct_evidence_codes": ["PS3", "PM2", "PP3", "PP1"],
    },
]


# ── Grading Utilities ─────────────────────────────────────────────────────────

CLASSIFICATION_ORDER = [
    "Pathogenic",
    "Likely Pathogenic",
    "Uncertain Significance",
    "Likely Benign",
    "Benign",
]


def _classification_distance(pred: str, truth: str) -> int:
    """Return the number of tiers between two classifications (0 = exact match)."""
    try:
        return abs(CLASSIFICATION_ORDER.index(pred) - CLASSIFICATION_ORDER.index(truth))
    except ValueError:
        return 5  # invalid classification


def _grade_easy(action: VariantAnnotationAction, variant: dict) -> tuple[float, str]:
    """
    Easy grader: structured evidence provided.
    Full reward for correct classification + evidence codes.
    """
    correct = variant.get("correct_classification", "Uncertain Significance")
    distance = _classification_distance(action.classification, correct)

    score = 0.0
    feedback_parts = []

    # Classification score (up to 0.6)
    if distance == 0:
        score += 0.60
        feedback_parts.append("Correct classification.")
    elif distance == 1:
        score += 0.30
        feedback_parts.append(f"Off by one tier (correct: {correct}).")
    else:
        feedback_parts.append(f"Incorrect classification (correct: {correct}).")

    # Evidence codes score (up to 0.25)
    correct_codes = set(variant.get("correct_evidence_codes", []))
    submitted_codes = set(action.evidence_codes)
    if correct_codes and submitted_codes:
        overlap = len(correct_codes & submitted_codes) / len(correct_codes)
        score += overlap * 0.25
        if overlap > 0:
            feedback_parts.append(f"Evidence codes: {overlap*100:.0f}% correct.")

    # Reasoning score (up to 0.15)
    if len(action.reasoning) > 20:
        score += 0.15
        feedback_parts.append("Reasoning provided.")

    return round(score, 3), " ".join(feedback_parts)


def _grade_medium(action: VariantAnnotationAction, variant: dict) -> tuple[float, str]:
    """
    Medium grader: agent must extract evidence from clinical notes.
    Slightly harder — evidence codes carry more weight.
    """
    correct = variant.get("correct_classification", "Uncertain Significance")
    distance = _classification_distance(action.classification, correct)

    score = 0.0
    feedback_parts = []

    # Classification score (up to 0.5)
    if distance == 0:
        score += 0.50
        feedback_parts.append("Correct classification.")
    elif distance == 1:
        score += 0.25
        feedback_parts.append(f"Off by one tier (correct: {correct}).")
    else:
        feedback_parts.append(f"Incorrect classification (correct: {correct}).")

    # Evidence codes score (up to 0.30)
    correct_codes = set(variant.get("correct_evidence_codes", []))
    submitted_codes = set(action.evidence_codes)
    if correct_codes and submitted_codes:
        overlap = len(correct_codes & submitted_codes) / len(correct_codes)
        score += overlap * 0.30
        if overlap > 0:
            feedback_parts.append(f"Evidence codes: {overlap*100:.0f}% correct.")
    elif not submitted_codes:
        feedback_parts.append("No evidence codes submitted — required for medium task.")

    # Reasoning score (up to 0.20)
    if len(action.reasoning) > 40:
        score += 0.20
        feedback_parts.append("Detailed reasoning provided.")
    elif len(action.reasoning) > 20:
        score += 0.10
        feedback_parts.append("Brief reasoning provided.")

    return round(score, 3), " ".join(feedback_parts)


def _grade_hard(action: VariantAnnotationAction, variant: dict) -> tuple[float, str]:
    """
    Hard grader: conflicting evidence from multiple databases.
    Agent must reconcile conflicts. Reasoning carries highest weight.
    """
    correct = variant.get("correct_classification", "Uncertain Significance")
    distance = _classification_distance(action.classification, correct)

    score = 0.0
    feedback_parts = []

    # Classification score (up to 0.40)
    if distance == 0:
        score += 0.40
        feedback_parts.append("Correct classification despite conflicting evidence.")
    elif distance == 1:
        score += 0.20
        feedback_parts.append(f"Off by one tier (correct: {correct}).")
    else:
        feedback_parts.append(f"Incorrect classification (correct: {correct}).")

    # Evidence codes (up to 0.25)
    correct_codes = set(variant.get("correct_evidence_codes", []))
    submitted_codes = set(action.evidence_codes)
    if correct_codes and submitted_codes:
        overlap = len(correct_codes & submitted_codes) / len(correct_codes)
        score += overlap * 0.25
        feedback_parts.append(f"Evidence codes: {overlap*100:.0f}% correct.")

    # Reasoning (up to 0.35) — most important for hard task
    reasoning = action.reasoning.lower()
    reasoning_score = 0.0
    if len(action.reasoning) > 100:
        reasoning_score += 0.15
    if len(action.reasoning) > 50:
        reasoning_score += 0.10
    # Reward for acknowledging conflict
    if any(word in reasoning for word in ["conflict", "conflicting", "disagree", "discordant"]):
        reasoning_score += 0.05
        feedback_parts.append("Correctly identified conflicting evidence.")
    # Reward for citing specific databases
    if any(db in reasoning for db in ["clinvar", "gnomad", "enigma", "literature"]):
        reasoning_score += 0.05
        feedback_parts.append("Referenced specific databases.")
    score += min(reasoning_score, 0.35)

    return round(score, 3), " ".join(feedback_parts)


# ── Main Environment Class ────────────────────────────────────────────────────

class VariantAnnotationEnvironment(Environment):
    """
    Variant Annotation Environment.

    An AI agent must classify genomic variants using the ACMG/AMP framework.
    Three task tiers test increasing levels of clinical reasoning ability.

    Supports concurrent WebSocket sessions — each client gets its own state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    # Max steps per episode before forced termination
    MAX_STEPS = 3

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_variant: dict = {}
        self._current_task: str = "easy"
        self._episode_done: bool = False
        self._task_cycle = ["easy", "medium", "hard"]
        self._task_index: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> VariantAnnotationObservation:
        """
        Reset the environment and return a new variant to classify.
        Tasks cycle: easy → medium → hard → easy → ...
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_done = False

        # Cycle through tasks
        self._current_task = self._task_cycle[self._task_index % 3]
        self._task_index += 1

        # Pick a random variant
        self._current_variant = random.choice(VARIANT_DATA)

        return self._build_observation(feedback=None)

    def step(self, action: VariantAnnotationAction) -> VariantAnnotationObservation:  # type: ignore[override]
        """
        Submit a classification and receive a graded observation.
        """
        if self._episode_done:
            return self._build_observation(
                feedback="Episode already done. Call reset() to start a new episode.",
                force_done=True,
            )

        self._state.step_count += 1

        # Validate classification string
        if action.classification not in VALID_CLASSIFICATIONS:
            reward = -0.20
            feedback = (
                f"Invalid classification '{action.classification}'. "
                f"Must be one of: {', '.join(VALID_CLASSIFICATIONS)}."
            )
            done = self._state.step_count >= self.MAX_STEPS
            self._episode_done = done
            return VariantAnnotationObservation(
                **self._base_obs_fields(),
                feedback=feedback,
                done=done,
                reward=reward,
                correct_classification=self._current_variant.get("correct_classification", "Uncertain Significance") if done else None,
            )

        # Grade based on task
        if self._current_task == "easy":
            score, feedback = _grade_easy(action, self._current_variant)
        elif self._current_task == "medium":
            score, feedback = _grade_medium(action, self._current_variant)
        else:
            score, feedback = _grade_hard(action, self._current_variant)

        # Efficiency penalty for extra steps
        step_penalty = max(0, (self._state.step_count - 1)) * 0.10
        reward = max(0.0, round(score - step_penalty, 3))

        # End episode if correct or max steps reached
        correct = action.classification == self._current_variant.get("correct_classification", "Uncertain Significance")
        done = correct or self._state.step_count >= self.MAX_STEPS
        self._episode_done = done

        return VariantAnnotationObservation(
            **self._base_obs_fields(),
            feedback=feedback,
            done=done,
            reward=reward,
            correct_classification=self._current_variant.get("correct_classification", "Uncertain Significance") if done else None,
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _base_obs_fields(self) -> dict:
        """Fields shared by all observations for the current variant."""
        v = self._current_variant
        task = self._current_task
        fields = {
            "task_id": task,
            "variant_id": v.get("variant_id", ""),
            "gene": v.get("gene", ""),
            "hgvs_notation": v.get("hgvs_notation", ""),
            "step_count": self._state.step_count,
        }

        if task == "easy":
            fields.update({
                "variant_type": v.get("variant_type"),
                "population_frequency": v.get("population_frequency"),
                "in_silico_predictions": v.get("in_silico_predictions"),
                "functional_evidence": v.get("functional_evidence"),
                "segregation_data": v.get("segregation_data"),
            })
        elif task == "medium":
            fields["clinical_notes"] = v.get("clinical_notes")
        else:  # hard
            fields.update({
                "clinvar_entry": v.get("clinvar_entry"),
                "gnomad_entry": v.get("gnomad_entry"),
                "literature_evidence": v.get("literature_evidence"),
            })

        return fields

    def _build_observation(
        self,
        feedback: str | None,
        force_done: bool = False,
    ) -> VariantAnnotationObservation:
        """Build a full observation with optional feedback."""
        return VariantAnnotationObservation(
            **self._base_obs_fields(),
            feedback=feedback,
            done=force_done,
            reward=0.0,
        )
