"""
Baseline Inference Script — Variant Annotation Environment
Meta PyTorch OpenEnv Hackathon x SST

The validator injects:
  API_BASE_URL  — LiteLLM proxy base URL for all LLM calls
  API_KEY       — API key for the LiteLLM proxy
  MODEL_NAME    — model to use (default: gpt-4o-mini)
  HF_TOKEN      — optional Hugging Face token
"""

import json
import os
import sys
import time
import requests
from openai import OpenAI

# ── Environment variables injected by validator ───────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL")          # LiteLLM proxy — NO default
API_KEY          = os.getenv("API_KEY")               # proxy API key — NO default
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI client — must use API_BASE_URL and API_KEY from environment ────────
if API_BASE_URL and API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    print(f"LLM client ready | base_url={API_BASE_URL} | model={MODEL_NAME}", file=sys.stderr)
else:
    client = None
    print("API_BASE_URL or API_KEY not set — using rule-based fallback", file=sys.stderr)

# ── OpenEnv environment URL ───────────────────────────────────────────────────
ENV_URL = "https://rahul06n-variant-annotation-env.hf.space"

TASKS = ["easy", "medium", "hard"]
NUM_EPISODES = 3

SYSTEM_PROMPT = """You are an expert clinical geneticist.
Classify genomic variants using the ACMG/AMP 5-tier system.
Choose exactly one of: Pathogenic, Likely Pathogenic, Uncertain Significance, Likely Benign, Benign.
Respond ONLY with valid JSON (no markdown, no extra text):
{"classification": "...", "evidence_codes": ["..."], "reasoning": "..."}"""

VALID = ["Pathogenic", "Likely Pathogenic", "Uncertain Significance", "Likely Benign", "Benign"]


# ── Rule-based fallback (no LLM needed) ──────────────────────────────────────

def rule_based(obs: dict) -> dict:
    task  = obs.get("task_id", "easy")

    if task == "easy":
        vtype = obs.get("variant_type", "") or ""
        freq  = obs.get("population_frequency") or 0.0
        preds = obs.get("in_silico_predictions") or {}
        func  = (obs.get("functional_evidence") or "").lower()
        seg   = (obs.get("segregation_data") or "").lower()
        sift  = str(preds.get("SIFT", "")).lower()
        poly  = str(preds.get("PolyPhen2", "")).lower()
        cadd  = float(preds.get("CADD", 0) or 0)

        if vtype in ("frameshift", "nonsense", "splice"):
            codes = ["PVS1", "PM2", "PP1"] if "segregates" in seg else ["PVS1", "PM2"]
            cls   = "Pathogenic" if freq < 0.001 else "Likely Pathogenic"
            return {"classification": cls, "evidence_codes": codes,
                    "reasoning": f"Null variant ({vtype}) — PVS1. AF={freq}."}

        if freq >= 0.05:
            return {"classification": "Benign", "evidence_codes": ["BA1", "BS1", "BP4"],
                    "reasoning": f"AF={freq:.1%} triggers BA1 stand-alone benign."}

        if freq >= 0.005 and ("normal" in func or "benign" in func or sift == "tolerated"):
            return {"classification": "Likely Benign", "evidence_codes": ["BS1", "BP4", "BS3"],
                    "reasoning": f"Common (AF={freq:.3%}) + benign functional evidence."}

        if freq < 0.0001 and (sift == "deleterious" or "damaging" in poly or cadd >= 20):
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PM2", "PP3"],
                    "reasoning": f"Rare (AF={freq}) + deleterious predictions."}

        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Insufficient evidence for definitive classification."}

    elif task == "medium":
        notes = (obs.get("clinical_notes") or "").lower()
        ps = sum(1 for s in ["frameshift","nonsense","truncat","loss of function",
                              "premature stop","absent from gnomad","pathogenic",
                              "co-segregates","segregates with"] if s in notes)
        bs = sum(1 for s in ["benign","polymorphism","normal function",
                              "does not segregate","healthy controls"] if s in notes)
        if ps >= 4:
            return {"classification": "Pathogenic", "evidence_codes": ["PVS1","PM2","PP1"],
                    "reasoning": "Strong pathogenic signals in clinical notes."}
        if ps >= 2:
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PM2","PP3"],
                    "reasoning": "Moderate pathogenic signals in clinical notes."}
        if bs >= 2 or "common" in notes:
            return {"classification": "Likely Benign", "evidence_codes": ["BS1","BP4"],
                    "reasoning": "Benign signals in clinical notes."}
        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Insufficient evidence in clinical notes."}

    else:  # hard
        clinvar    = obs.get("clinvar_entry") or {}
        gnomad     = obs.get("gnomad_entry") or {}
        af         = gnomad.get("allele_frequency") or 0.0
        submitters = clinvar.get("submitters") or []

        enigma_cls = next((s.get("classification","") for s in submitters
                           if "enigma" in s.get("name","").lower()), None)
        if enigma_cls:
            if "likely pathogenic" in enigma_cls.lower():
                return {"classification": "Likely Pathogenic",
                        "evidence_codes": ["PS3","PM2","PP3","PP1"],
                        "reasoning": f"ENIGMA expert panel: {enigma_cls}."}
            if "pathogenic" in enigma_cls.lower():
                return {"classification": "Pathogenic",
                        "evidence_codes": ["PS3","PM2","PP1"],
                        "reasoning": "ENIGMA expert panel Pathogenic."}
            if "benign" in enigma_cls.lower():
                cls = "Benign" if "likely" not in enigma_cls.lower() else "Likely Benign"
                return {"classification": cls,
                        "evidence_codes": ["BA1" if af > 0.05 else "BS1","BP4"],
                        "reasoning": f"ENIGMA expert panel: {enigma_cls}."}

        if af >= 0.05:
            return {"classification": "Benign", "evidence_codes": ["BA1","BS1"],
                    "reasoning": f"AF={af:.1%} triggers BA1."}
        if af >= 0.005:
            return {"classification": "Likely Benign", "evidence_codes": ["BS1","BP4"],
                    "reasoning": f"AF={af:.3%} too common for pathogenic variant."}

        path_votes = sum(1 for s in submitters
                         if "pathogenic" in s.get("classification","").lower()
                         and "benign" not in s.get("classification","").lower())
        if path_votes > 0 and af < 0.001:
            return {"classification": "Likely Pathogenic", "evidence_codes": ["PM2","PP3"],
                    "reasoning": "ClinVar pathogenic majority + extremely rare."}

        return {"classification": "Uncertain Significance", "evidence_codes": ["PM2"],
                "reasoning": "Conflicting evidence — insufficient consensus."}


# ── LLM classify via validator proxy ─────────────────────────────────────────

def build_prompt(obs: dict) -> str:
    task = obs.get("task_id", "easy")
    p = f"Gene: {obs.get('gene')}\nHGVS: {obs.get('hgvs_notation')}\nTask: {task}\n\n"
    if task == "easy":
        p += f"Variant type: {obs.get('variant_type')}\n"
        p += f"Population AF: {obs.get('population_frequency')}\n"
        for k, v in (obs.get("in_silico_predictions") or {}).items():
            p += f"{k}: {v}\n"
        p += f"Functional: {obs.get('functional_evidence')}\n"
        p += f"Segregation: {obs.get('segregation_data')}\n"
    elif task == "medium":
        p += f"Clinical notes:\n{obs.get('clinical_notes')}\n"
    else:
        cv = obs.get("clinvar_entry") or {}
        for s in cv.get("submitters", []):
            p += f"ClinVar {s['name']}: {s['classification']}\n"
        gn = obs.get("gnomad_entry") or {}
        p += f"gnomAD AF: {gn.get('allele_frequency')}\n"
        for lit in (obs.get("literature_evidence") or []):
            p += f"PMID {lit.get('pmid')}: {lit.get('classification')}\n"
    p += "\nRespond with JSON only."
    return p


def llm_classify(obs: dict) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        if result.get("classification") not in VALID:
            raise ValueError(f"Invalid: {result.get('classification')}")
        return result
    except Exception as e:
        print(f"LLM error: {e} — rule-based fallback", file=sys.stderr)
        return rule_based(obs)


# ── Episode ───────────────────────────────────────────────────────────────────

def run_episode(env_url: str) -> dict:
    task   = "unknown"
    reward = 0.0

    try:
        r = requests.post(f"{env_url}/reset", timeout=30)
        r.raise_for_status()
        obs  = r.json()["observation"]
        task = obs["task_id"]
        print(f"Task: {task} | {obs.get('variant_id','')}", file=sys.stderr)
    except Exception as e:
        print(f"Reset error: {e}", file=sys.stderr)
        sys.stdout.write(f"[START] task={task}\n");                      sys.stdout.flush()
        sys.stdout.write(f"[STEP] step=1 reward=0.0\n");                 sys.stdout.flush()
        sys.stdout.write(f"[END] task={task} score=0.0 steps=1\n");      sys.stdout.flush()
        return {"task": task, "reward": 0.0}

    try:
        action = llm_classify(obs) if client else rule_based(obs)
        print(f"-> {action.get('classification')}", file=sys.stderr)
    except Exception as e:
        print(f"Classify error: {e}", file=sys.stderr)
        action = {"classification": "Uncertain Significance",
                  "evidence_codes": [], "reasoning": "Error."}

    try:
        r2 = requests.post(f"{env_url}/step",
                           json={"action": {**action, "metadata": {}}},
                           timeout=30)
        r2.raise_for_status()
        d      = r2.json()
        reward = float(d.get("reward", 0.0))
        print(f"Reward: {reward:.3f} | {d['observation'].get('feedback','')}", file=sys.stderr)
    except Exception as e:
        print(f"Step error: {e}", file=sys.stderr)

    safe_score = round(max(0.01, min(0.99, reward)), 3)
    sys.stdout.write(f"[START] task={task}\n");                               sys.stdout.flush()
    sys.stdout.write(f"[STEP] step=1 reward={safe_score}\n");                 sys.stdout.flush()
    sys.stdout.write(f"[END] task={task} score={safe_score} steps=1\n");      sys.stdout.flush()

    return {"task": task, "reward": reward}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"API_BASE_URL : {API_BASE_URL}", file=sys.stderr)
    print(f"MODEL_NAME   : {MODEL_NAME}",   file=sys.stderr)
    print(f"ENV_URL      : {ENV_URL}",       file=sys.stderr)
    print(f"Agent        : {'LLM via proxy' if client else 'rule-based'}", file=sys.stderr)

    try:
        h = requests.get(f"{ENV_URL}/health", timeout=30)
        print(f"Health: {h.json()}", file=sys.stderr)
    except Exception as e:
        print(f"Health warning: {e}", file=sys.stderr)

    task_scores: dict = {"easy": [], "medium": [], "hard": []}
    total = NUM_EPISODES * len(TASKS)

    for ep in range(1, total + 1):
        print(f"\n[Episode {ep}/{total}]", file=sys.stderr)
        try:
            res = run_episode(ENV_URL)
            t   = res.get("task", "unknown")
            if t in task_scores:
                task_scores[t].append(res.get("reward", 0.0))
        except Exception as e:
            print(f"Episode {ep} error: {e}", file=sys.stderr)
        time.sleep(0.5)

    print("\n=== FINAL RESULTS ===", file=sys.stderr)
    overall = []
    for t in TASKS:
        s = task_scores[t]
        if s:
            overall.extend(s)
            print(f"{t.upper()}: avg={sum(s)/len(s):.3f} min={min(s):.3f} max={max(s):.3f}",
                  file=sys.stderr)
    if overall:
        print(f"OVERALL AVG: {sum(overall)/len(overall):.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
