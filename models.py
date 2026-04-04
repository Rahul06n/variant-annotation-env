# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Data models for the Variant Annotation Environment.

A real-world bioinformatics environment where an AI agent classifies
genomic variants using the ACMG/AMP classification framework.

Tasks:
  - Task 1 (Easy):   Classify a variant given pre-extracted structured evidence
  - Task 2 (Medium): Classify a variant from raw clinical notes
  - Task 3 (Hard):   Reconcile conflicting evidence from multiple databases
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── ACMG Classification Categories ──────────────────────────────────────────

VALID_CLASSIFICATIONS = [
    "Pathogenic",
    "Likely Pathogenic",
    "Uncertain Significance",
    "Likely Benign",
    "Benign",
]

VALID_TASKS = ["easy", "medium", "hard"]


# ── Action ───────────────────────────────────────────────────────────────────

class VariantAnnotationAction(Action):
    """
    Action submitted by the agent to classify a genomic variant.

    The agent must:
      1. Choose a classification from the ACMG 5-tier system
      2. List the evidence criteria it used (e.g. ["PS1", "PM2", "BP4"])
      3. Provide a short reasoning string explaining its decision
    """

    classification: str = Field(
        ...,
        description=(
            "ACMG classification. Must be one of: "
            "'Pathogenic', 'Likely Pathogenic', 'Uncertain Significance', "
            "'Likely Benign', 'Benign'."
        ),
    )

    evidence_codes: list[str] = Field(
        default_factory=list,
        description=(
            "ACMG/AMP evidence codes used (e.g. ['PS1', 'PM2', 'BP4']). "
            "At least one code is required for medium/hard tasks."
        ),
    )

    reasoning: str = Field(
        default="",
        description="Short explanation of why this classification was chosen.",
    )


# ── Observation ──────────────────────────────────────────────────────────────

class VariantAnnotationObservation(Observation):
    """
    Observation returned to the agent describing a variant to classify.

    Easy task:   structured evidence already extracted — just classify.
    Medium task: raw clinical notes provided — agent must extract evidence.
    Hard task:   conflicting entries from ClinVar, gnomAD, literature — reconcile.
    """

    # ── Which task is active ─────────────────────────────────────────────────
    task_id: str = Field(
        default="easy",
        description="Current task difficulty: 'easy', 'medium', or 'hard'.",
    )

    # ── Core variant identity ────────────────────────────────────────────────
    variant_id: str = Field(
        default="",
        description="Variant identifier, e.g. 'NM_007294.3:c.5266dupC'.",
    )

    gene: str = Field(
        default="",
        description="Gene symbol, e.g. 'BRCA1'.",
    )

    hgvs_notation: str = Field(
        default="",
        description="HGVS notation of the variant.",
    )

    # ── Easy task fields (structured evidence) ───────────────────────────────
    variant_type: Optional[str] = Field(
        default=None,
        description="Variant type: 'missense', 'nonsense', 'frameshift', 'splice', 'synonymous'.",
    )

    population_frequency: Optional[float] = Field(
        default=None,
        description="Allele frequency in gnomAD (0.0 = absent, 1.0 = fixed).",
    )

    in_silico_predictions: Optional[dict] = Field(
        default=None,
        description="In-silico tool predictions, e.g. {'SIFT': 'deleterious', 'PolyPhen2': 'probably_damaging'}.",
    )

    functional_evidence: Optional[str] = Field(
        default=None,
        description="Summary of functional study results if available.",
    )

    segregation_data: Optional[str] = Field(
        default=None,
        description="Segregation data from family studies if available.",
    )

    # ── Medium task fields (raw clinical notes) ──────────────────────────────
    clinical_notes: Optional[str] = Field(
        default=None,
        description=(
            "Raw clinical notes for medium task. "
            "Agent must extract relevant evidence before classifying."
        ),
    )

    # ── Hard task fields (conflicting database entries) ──────────────────────
    clinvar_entry: Optional[dict] = Field(
        default=None,
        description="ClinVar database entry with submitter classifications.",
    )

    gnomad_entry: Optional[dict] = Field(
        default=None,
        description="gnomAD population frequency data.",
    )

    literature_evidence: Optional[list[dict]] = Field(
        default=None,
        description="List of relevant literature entries with conflicting findings.",
    )

    # ── Feedback after a step ────────────────────────────────────────────────
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback from the grader after a classification attempt.",
    )

    correct_classification: Optional[str] = Field(
        default=None,
        description="Revealed after episode ends (done=True).",
    )

    step_count: int = Field(
        default=0,
        description="Number of steps taken in this episode.",
    )
