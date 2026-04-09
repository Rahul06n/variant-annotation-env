"""
Baseline Inference Script — Variant Annotation Environment
Follows the Meta PyTorch OpenEnv Hackathon x SST sample inference.py format exactly.
"""

import argparse
import json
import os
import sys
import time
import requests
from openai import OpenAI

# ── Required environment variables (per checklist) ────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://rahul06n-variant-annotation-env.hf.space")
MODEL_NAME   = os.getenv("MODEL_NAME", "rule-based")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI client configured via environment variables ────────────────────────
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_KEY:
    client = OpenAI(
        api_key=GEMINI_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
elif OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
else:
    client = None
    MODEL_NAME = os.getenv("MODEL_NAME", "rule-based")

TASKS = ["easy", "medium", "hard"]
NUM_EPISODES_PER_TASK = 3

SYSTEM_PROMPT = """You are an expert clinical geneticist.
Classify genomic variants using the ACMG/AMP 5-tier system.
Choose exactly one: Pathogenic, Likely Pathogenic, Uncertain Significance, Likely Benign, Benign.
Respond ONLY with valid JSON:
{"classification": "...", "evidence_codes": ["..."], "reasoning": "..."}"""


# ── Rule-based fallback agent ─────────────────────────────────────────────────

def rule_based_agent(observation: dict) -> dict:
    task  = observation.get("task_id", "easy")

    if task == "easy":
        vtype = observation.get("variant_type", "") or ""
        freq  = observation.get("population_frequency") or 0.0
        preds = observation.get("in_silico_predictions") or {}
        func  = (observation.get("functional_evidence") or "").lower()
        seg   = (observation.get("segregation_data") or "").lower()
        sift  = str(preds.get("SIFT", "")).lower()
        poly  = str(preds.get("PolyPhen2", "")).lower()
        cadd  = float(preds.get("CADD", 0) or 0)

        if vtype in ("frameshift", "nonsense", "splice"):
            if freq < 0.001:
                codes = ["PVS1", "PM2", "PP1"] if "segregates" in seg else ["PVS1", "PM2"]
                return {"classification": "Pathogenic", "evidence_codes": codes,
                        "reasoning": f"Null variant ({vtype}) triggers PVS1. Extremely rare AF={freq}."}
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PVS1"],
                    "reasoning": "Frameshift but population frequency higher than expected."}

        if freq >= 0.05:
            return {"classification": "Benign", "evidence_codes": ["BA1", "BS1", "BP4"],
                    "reasoning": f"AF={freq:.1%} exceeds 5% — BA1 stand-alone benign criterion."}

        if freq >= 0.005:
            if "normal" in func or "benign" in func or sift == "tolerated":
                return {"classification": "Likely Benign", "evidence_codes": ["BS1", "BP4", "BS3"],
                        "reasoning": f"Common AF={freq:.3%} with benign functional evidence."}

        if freq < 0.0001 and (sift == "deleterious" or "damaging" in poly or cadd >= 20):
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PM2", "PP3"],
                    "reasoning": f"Rare AF={freq} with deleterious in-silico predictions."}

        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Insufficient evidence to classify as pathogenic or benign."}

    elif task == "medium":
        notes = (observation.get("clinical_notes") or "").lower()
        path_signals  = ["frameshift", "nonsense", "truncat", "null allele", "loss of function",
                         "premature stop", "absent from gnomad", "pathogenic", "loss of brct",
                         "loss of rad51", "co-segregates", "segregates with"]
        benign_signals = ["benign", "polymorphism", "tolerated", "normal function",
                          "normal protein", "does not segregate", "healthy controls"]
        ps = sum(1 for s in path_signals  if s in notes)
        bs = sum(1 for s in benign_signals if s in notes)

        if ps >= 4:
            return {"classification": "Pathogenic", "evidence_codes": ["PVS1", "PM2", "PP1"],
                    "reasoning": "Clinical notes: null variant, absent from population, segregation confirmed."}
        elif ps >= 2:
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PM2", "PP3"],
                    "reasoning": "Clinical notes suggest pathogenic features."}
        elif bs >= 3 or "common" in notes:
            return {"classification": "Likely Benign", "evidence_codes": ["BS1", "BP4"],
                    "reasoning": "Clinical notes indicate benign features."}
        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Insufficient evidence in clinical notes."}

    else:  # hard
        clinvar    = observation.get("clinvar_entry") or {}
        gnomad     = observation.get("gnomad_entry") or {}
        af         = gnomad.get("allele_frequency") or 0.0
        submitters = clinvar.get("submitters") or []

        enigma = next((s.get("classification", "") for s in submitters
                       if "enigma" in s.get("name", "").lower()), None)
        if enigma:
            if "likely pathogenic" in enigma.lower():
                return {"classification": "Likely Pathogenic",
                        "evidence_codes": ["PS3", "PM2", "PP3", "PP1"],
                        "reasoning": f"ENIGMA expert panel: {enigma}."}
            elif "pathogenic" in enigma.lower():
                return {"classification": "Pathogenic",
                        "evidence_codes": ["PS3", "PM2", "PP1"],
                        "reasoning": "ENIGMA expert panel Pathogenic."}
            elif "benign" in enigma.lower():
                cls = "Benign" if "likely" not in enigma.lower() else "Likely Benign"
                return {"classification": cls,
                        "evidence_codes": ["BA1" if af > 0.05 else "BS1", "BP4"],
                        "reasoning": f"ENIGMA expert panel: {enigma}."}

        if af >= 0.05:
            return {"classification": "Benign", "evidence_codes": ["BA1", "BS1"],
                    "reasoning": f"AF={af:.1%} triggers BA1."}
        if af >= 0.005:
            return {"classification": "Likely Benign", "evidence_codes": ["BS1", "BP4"],
                    "reasoning": f"AF={af:.3%} too common for high-penetrance variant."}

        path_votes = sum(1 for s in submitters
                         if "pathogenic" in s.get("classification", "").lower()
                         and "benign" not in s.get("classification", "").lower())
        if path_votes > 0 and af < 0.001:
            return {"classification": "Likely Pathogenic",
                    "evidence_codes": ["PM2", "PP3"],
                    "reasoning": "ClinVar majority pathogenic + extremely rare."}

        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Conflicting evidence — insufficient consensus."}


# ── LLM agent ─────────────────────────────────────────────────────────────────

def build_prompt(observation: dict) -> str:
    task = observation.get("task_id", "easy")
    prompt = f"Gene: {observation.get('gene')}\nHGVS: {observation.get('hgvs_notation')}\nTask: {task}\n\n"
    if task == "easy":
        prompt += f"Type: {observation.get('variant_type')}\nAF: {observation.get('population_frequency')}\n"
        for k, v in (observation.get("in_silico_predictions") or {}).items():
            prompt += f"{k}: {v}\n"
        prompt += f"Functional: {observation.get('functional_evidence')}\n"
        prompt += f"Segregation: {observation.get('segregation_data')}\n"
    elif task == "medium":
        prompt += f"Clinical notes:\n{observation.get('clinical_notes')}\n"
    else:
        cv = observation.get("clinvar_entry") or {}
        for s in cv.get("submitters", []):
            prompt += f"ClinVar {s['name']}: {s['classification']}\n"
        gn = observation.get("gnomad_entry") or {}
        prompt += f"gnomAD AF: {gn.get('allele_frequency')}\n"
        for p in (observation.get("literature_evidence") or []):
            prompt += f"PMID {p.get('pmid')}: {p.get('classification')}\n"
    return prompt


def llm_classify(observation: dict) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(observation)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        valid = ["Pathogenic", "Likely Pathogenic", "Uncertain Significance",
                 "Likely Benign", "Benign"]
        if result.get("classification") not in valid:
            raise ValueError(f"Invalid classification: {result.get('classification')}")
        return result
    except Exception as e:
        print(f"LLM error: {e} — using rule-based fallback", file=sys.stderr)
        return rule_based_agent(observation)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(base_url: str) -> dict:
    task   = "unknown"
    reward = 0.0

    try:
        reset_resp = requests.post(f"{base_url}/reset", timeout=30)
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]
        task        = observation["task_id"]
        print(f"Task: {task} | Variant: {observation.get('variant_id', 'N/A')}", file=sys.stderr)
    except Exception as e:
        print(f"Reset failed: {e}", file=sys.stderr)
        sys.stdout.write(f"[START] task={task}\n"); sys.stdout.flush()
        sys.stdout.write(f"[STEP] step=1 reward=0.0\n"); sys.stdout.flush()
        sys.stdout.write(f"[END] task={task} score=0.0 steps=1\n"); sys.stdout.flush()
        return {"task": task, "reward": 0.0}

    try:
        action = llm_classify(observation) if client else rule_based_agent(observation)
        print(f"Classification: {action.get('classification')}", file=sys.stderr)
    except Exception as e:
        print(f"Classification error: {e}", file=sys.stderr)
        action = {"classification": "Uncertain Significance",
                  "evidence_codes": [], "reasoning": "Error."}

    try:
        step_resp = requests.post(
            f"{base_url}/step",
            json={"action": {**action, "metadata": {}}},
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()
        reward    = float(step_data.get("reward", 0.0))
        feedback  = step_data["observation"].get("feedback", "")
        print(f"Reward: {reward:.3f} | {feedback}", file=sys.stderr)
    except Exception as e:
        print(f"Step failed: {e}", file=sys.stderr)

    # ── Required structured stdout output ──────────────────────────────────
    sys.stdout.write(f"[START] task={task}\n");                    sys.stdout.flush()
    sys.stdout.write(f"[STEP] step=1 reward={round(reward,3)}\n"); sys.stdout.flush()
    sys.stdout.write(f"[END] task={task} score={round(reward,3)} steps=1\n"); sys.stdout.flush()

    return {"task": task, "reward": reward,
            "classification": action.get("classification", "")}


# ── Main ───────────────────────────────────────────────────────────────────────

def run_baseline(base_url: str, num_episodes: int = NUM_EPISODES_PER_TASK) -> None:
    print(f"API_BASE_URL : {base_url}",                          file=sys.stderr)
    print(f"MODEL_NAME   : {MODEL_NAME}",                        file=sys.stderr)
    print(f"Agent        : {'LLM' if client else 'rule-based'}", file=sys.stderr)

    try:
        health = requests.get(f"{base_url}/health", timeout=30)
        print(f"Health: {health.json()}", file=sys.stderr)
    except Exception as e:
        print(f"Health check warning: {e}", file=sys.stderr)

    task_scores: dict = {"easy": [], "medium": [], "hard": []}
    total = num_episodes * len(TASKS)

    for ep in range(1, total + 1):
        print(f"\n[Episode {ep}/{total}]", file=sys.stderr)
        try:
            result = run_episode(base_url)
            t = result.get("task", "unknown")
            if t in task_scores:
                task_scores[t].append(result.get("reward", 0.0))
        except Exception as e:
            print(f"Episode {ep} failed: {e}", file=sys.stderr)
        time.sleep(0.5)

    print("\n=== RESULTS ===", file=sys.stderr)
    overall = []
    for t in TASKS:
        s = task_scores[t]
        if s:
            avg = sum(s) / len(s)
            overall.extend(s)
            print(f"{t.upper()}: avg={avg:.3f} min={min(s):.3f} max={max(s):.3f}",
                  file=sys.stderr)
    if overall:
        print(f"OVERALL: {sum(overall)/len(overall):.3f}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      default=API_BASE_URL)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_PER_TASK)
    args = parser.parse_args()
    run_baseline(base_url=args.url, num_episodes=args.episodes)
