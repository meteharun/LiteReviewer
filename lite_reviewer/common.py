from __future__ import annotations

import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

# =========================
# Project-relative paths
# =========================

# <repo_root>/lite_reviewer/common.py  -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "prompts"
TOKEN_PATH = REPO_ROOT / "github_token.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Deterministic, human-readable filenames
def repo_key(owner_repo: str, pr_number: int) -> str:
    owner, repo = owner_repo.split("/", 1)
    return f"{owner}__{repo}__pr{pr_number}"

def diff_path(owner_repo: str, pr_number: int) -> Path:
    return DATA_DIR / f"{repo_key(owner_repo, pr_number)}__diff.jsonl"

def reviews_path(owner_repo: str, pr_number: int, model_id: str, shot: str) -> Path:
    # model_id is the actual engine name (e.g., "phi3:mini"); normalize filename-friendly
    safe_model = model_id.replace(":", "_")
    return DATA_DIR / f"{repo_key(owner_repo, pr_number)}__reviews__{safe_model}__{shot}.jsonl"


# =========================
# I/O helpers
# =========================

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed rows silently
                continue

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# Prompts
# =========================

def load_prompt(name: str) -> str:
    """
    name ∈ {"zero_shot.json", "few_shot.json"} or a custom filename in /prompts.
    Returns the prompt string from {"prompt": "..."} JSON.
    """
    p = PROMPTS_DIR / name
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "prompt" in obj and isinstance(obj["prompt"], str):
        return obj["prompt"]
    raise ValueError(f"Prompt file must be an object with key 'prompt': {p}")


# =========================
# Model selection
# =========================

MODEL_MAP = {
    "phi": "phi3:mini",
    "mistral": "mistral:7b-instruct",
    "gemma": "gemma2:latest",
}

def resolve_model(key: str) -> str:
    key = key.lower().strip()
    if key not in MODEL_MAP:
        raise ValueError(f"Unknown model key '{key}'. Choose from: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[key]


# =========================
# Ollama client
# =========================

OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434/api/generate"

def ollama_generate(prompt: str, model: str, timeout_s: int = 180, **opt_overrides) -> str:
    """
    Calls Ollama /api/generate (non-stream) and returns the 'response' text.
    - Forces CPU (num_gpu=0)
    - Conservative memory/context
    - Retries with smaller context on OOM-style errors
    - format='json' to bias valid JSON output
    """
    url = os.getenv("OLLAMA_URL", OLLAMA_URL_DEFAULT)

    # Baseline: deterministic + CPU-only
    options = {
        "temperature": 0.0,   # reduce drift
        "num_predict": 200,
        "num_ctx": 2048,      # slightly higher than 1024 to avoid truncating few-shot + diff
        "num_batch": 1,
        "num_gpu": 0,
        "seed": 7,
    }
    # Gemma can be heavier; keep context modest
    if model.startswith("gemma2:"):
        options.update({"num_ctx": 1536})

    if opt_overrides:
        options.update(opt_overrides)

    def _call(opts):
        payload = {
            "model": model,
            "prompt": prompt,
            "options": opts,
            "stream": False,
            "format": "json",
        }
        r = requests.post(url, json=payload, timeout=timeout_s)
        # If server returns non-JSON (rare), still raise cleanly
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            raise
        if r.status_code >= 400 or ("error" in data):
            raise RuntimeError(f"Ollama error ({r.status_code}): {data.get('error') or data}")
        return (data.get("response") or "").strip()

    try:
        return _call(options)
    except RuntimeError as e:
        msg = str(e).lower()
        mem_triggers = (
            "requires more system memory",
            "unable to load full model on gpu",
            "out of memory",
            "not enough memory",
        )
        if any(t in msg for t in mem_triggers):
            tiny_opts = dict(options)
            tiny_opts.update({
                "num_ctx": 1024,
                "num_predict": 160,
                "num_batch": 1,
                "num_gpu": 0,
            })
            time.sleep(0.2)
            return _call(tiny_opts)
        raise


# =========================
# JSON repair (LLM outputs)
# =========================

# Heuristic fixer for broken JSON lists of objects like:
# [ { "line": 3, "type": "STYLE", "comment": "..." }, ... ]
_JSON_TYPE_FIX = re.compile(r'"type([a-zA-Z0-9\s])')  # e.g., "typeSome text

def repair_json_string(s: str) -> str:
    s = s.strip().strip("`")
    # isolate first array if the model wrapped with prose
    if "[" in s:
        s = s[s.find("[") :]
    # Attempt to fix broken "type..." keys:
    s = _JSON_TYPE_FIX.sub(r'"type": "OTHER", "comment": "\1', s)
    # Ensure objects separated by commas
    s = s.replace('}\n  {', '},\n  {')
    # Close comment strings before object end (very heuristic)
    s = s.replace('\n  }', '",\n  }')
    return s

def _coerce_single_object_to_array(obj: dict) -> Optional[List[Dict[str, Any]]]:
    # Accept if it contains at least "comment" (line/type are optional for hunk-wide posting)
    if "comment" in obj:
        return [obj]
    return None

def parse_model_json(s: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a model response into a list[dict]. Accept:
      - proper JSON array
      - single JSON object (wrap to list) if it has at least 'comment'
    """
    s = s.strip()

    # strip code fences if present
    if s.startswith("```"):
        s = s.strip("`")
        i = s.find("[")
        if i != -1:
            s = s[i:]

    # 1) Try direct parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return _coerce_single_object_to_array(parsed)
    except Exception:
        pass

    # 2) Try repaired string
    try:
        fixed = repair_json_string(s)
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return _coerce_single_object_to_array(parsed)
    except Exception:
        pass

    # 3) Last attempt: extract the first [...] slice
    lb = s.find("[")
    rb = s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            chunk = s[lb : rb + 1]
            parsed = json.loads(chunk)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return None


# =========================
# Prompt filling
# =========================

def fill_prompt(template: str, context: str, diff_hunk: str) -> str:
    return template.replace("{{context}}", context or "").replace("{{diff_hunk}}", diff_hunk or "")


# =========================
# GitHub helpers
# =========================

API_BASE = "https://api.github.com"

def load_github_token() -> Optional[str]:
    # 1) Env var wins
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token.strip()
    # 2) Fallback to file in repo root
    if TOKEN_PATH.exists():
        txt = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return None

def gh_headers(token: Optional[str]) -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LiteReviewer/1.0",
    }
    if token:
        h["Authorization"] = f"token {token}"
    return h

def gh_get(url: str, token: str | None, params=None):
    r = requests.get(url, headers=gh_headers(token), params=params, timeout=30)
    r.raise_for_status()
    return r

def gh_post(url: str, token: str | None, json_body: dict):
    r = requests.post(url, headers=gh_headers(token), json=json_body, timeout=30)
    return r


# =========================
# Position selection for posting
# =========================

def choose_position_for_hunk(hunk: Dict[str, Any], file_position_table: Dict[str, int]) -> Optional[int]:
    """
    Strategy:
    - Prefer the first '+' line's *new* line number in the hunk; map to unified 'position' via position_table.
    - If no '+' lines, fall back to the first context (' ') line in the hunk.
    - If still nothing, return None.
    """
    # 1) first plus line
    for ln in hunk.get("lines", []):
        if ln.get("tag") == "+":
            new_no = ln.get("new")
            if isinstance(new_no, int):
                pos = file_position_table.get(str(new_no)) or file_position_table.get(new_no)
                if isinstance(pos, int):
                    return pos
    # 2) first neutral context line
    for ln in hunk.get("lines", []):
        if ln.get("tag") == " ":
            new_no = ln.get("new")
            if isinstance(new_no, int):
                pos = file_position_table.get(str(new_no)) or file_position_table.get(new_no)
                if isinstance(pos, int):
                    return pos
    return None


# =========================
# Lightweight logger
# =========================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def log_err(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


# =========================
# Small utilities
# =========================

def now_ts() -> int:
    return int(time.time())
