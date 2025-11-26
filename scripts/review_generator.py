import os
import sys
import json
import re
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# =================== CONFIG (adjust paths) ===================
PROMPT_ZERO_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\prompts\zero_shot.json"
PROMPT_FEW_PATH  = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\prompts\few_shot.json"
OLLAMA_URL       = "http://127.0.0.1:11434/api/generate"
DEFAULT_OUT_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\data\generated_reviews.jsonl"
DEFAULT_TIMEOUT  = 180

MODEL_MAP = {
    "phi": "phi3:mini",
    "mistral": "mistral:7b-instruct",
    "gemma": "gemma2:latest",
}
# ============================================================


# ---------- Robust JSON Parser Utilities ----------
ALLOWED_TYPES = {"STYLE", "LOGIC", "DOCUMENTATION", "SECURITY", "PERFORMANCE", "OTHER"}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        i = s.find("[")
        if i != -1:
            s = s[i:]
    return s

def _isolate_first_json_array(s: str) -> str:
    start = s.find("[")
    if start == -1:
        return s
    end = s.rfind("]")
    if end == -1 or end <= start:
        return s[start:]
    return s[start:end+1]

def _fix_type_concat_bug(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        j = s.find('"type', i)
        if j == -1:
            out.append(s[i:])
            break

        out.append(s[i:j])
        k = j + len('"type')
        if k < len(s) and s[k] != '"':
            end_candidates = []
            for pat in [r'}\s*,', r'}\s*\]?', r'}\s*\n']:
                m = re.search(pat, s[k:], flags=re.DOTALL)
                if m:
                    end_candidates.append(k + m.start())
            obj_end = min(end_candidates) if end_candidates else len(s)-1
            accidental = s[k:obj_end].strip()
            replacement = '"type": "OTHER", "comment": ' + json.dumps(accidental)
            out.append(replacement)
            i = obj_end
            continue
        else:
            out.append(s[j:j+6])  # '"type"'
            i = j + 6
    return "".join(out)

def _normalize_types(objs: List[Dict]) -> List[Dict]:
    norm = []
    for o in objs:
        line = o.get("line")
        comment = o.get("comment")
        ctype = o.get("type", "OTHER")
        if isinstance(ctype, str):
            ctype_up = ctype.strip().upper()
            ctype = ctype_up if ctype_up in ALLOWED_TYPES else "OTHER"
        else:
            ctype = "OTHER"
        norm.append({"line": line, "type": ctype, "comment": comment})
    return norm

def parse_model_json_strict(raw: str) -> Tuple[List[Dict], bool]:
    """Returns (comments, parse_fail)"""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return _normalize_types(data), False
    except Exception:
        pass

    s = _strip_code_fences(raw)
    s = _isolate_first_json_array(s)

    try:
        data = json.loads(s)
        if isinstance(data, list):
            return _normalize_types(data), False
    except Exception:
        pass

    s2 = _fix_type_concat_bug(s)
    try:
        data = json.loads(s2)
        if isinstance(data, list):
            return _normalize_types(data), False
    except Exception:
        return [], True

    return [], True
# -------------------------------------------------


def load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "prompt" in obj:
        return obj["prompt"]
    raise ValueError(f"Prompt file must be a JSON object with key 'prompt': {path}")


def fill_prompt(template: str, context: str, diff_hunk: str) -> str:
    return template.replace("{{context}}", context or "").replace("{{diff_hunk}}", diff_hunk or "")


def ask_ollama(prompt: str, model: str, timeout_s: int = DEFAULT_TIMEOUT) -> str:
    def _call(opts):
        payload = {"model": model, "prompt": prompt, "options": opts, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        data = r.json()
        if r.status_code >= 400 or "error" in data:
            raise RuntimeError(f"Ollama error ({r.status_code}): {data.get('error') or data}")
        return data.get("response", "").strip()

    # first attempt (normal)
    base_opts = {"temperature": 0.2, "num_predict": 256, "num_ctx": 2048}
    try:
        return _call(base_opts)
    except RuntimeError as e:
        msg = str(e).lower()
        # fallback on memory/GPU errors
        if "requires more system memory" in msg or "unable to load full model" in msg or "gpu" in msg:
            fallback = {"temperature": 0.2, "num_predict": 128, "num_ctx": 1024, "num_gpu": 0, "low_vram": True}
            return _call(fallback)
        raise



def build_context_from_hunk(hunk: Dict[str, Any], max_lines: int = 200) -> str:
    neutral = []
    for entry in hunk.get("lines", []):
        tag = entry.get("tag")
        text = entry.get("text", "")
        if tag == " ":
            neutral.append(text)
    if len(neutral) > max_lines:
        head = neutral[: max_lines // 2]
        tail = neutral[-(max_lines - len(head)) :]
        return "\n".join(head + ["..."] + tail)
    return "\n".join(neutral)


def build_diff_from_hunk(hunk: Dict[str, Any]) -> str:
    header = hunk.get("header") or f"@@ -{hunk.get('old_start', '?')},{hunk.get('old_len', '?')} +{hunk.get('new_start', '?')},{hunk.get('new_len', '?')} @@"
    out = [header]
    for entry in hunk.get("lines", []):
        tag = entry.get("tag")
        text = entry.get("text", "")
        if tag == "+":
            out.append("+" + text)
        elif tag == "-":
            out.append("-" + text)
    return "\n".join(out)


def absolute_new_line(hunk: dict, approx_line: object) -> int | None:
    """
    Robustly map model 'line' → absolute NEW-file line inside this hunk.
    Tries, in order:
      A) 'line' is 1-based index over ONLY +/- lines shown to the model.
      B) 'line' is an absolute new-file line number (if within hunk range).
      C) 'line' is offset from new_start (new_start + line - 1).
      D) Fallback: nearest '+' line in the hunk.
    Returns None if no '+' exists to anchor.
    """
    if not isinstance(approx_line, int) or approx_line < 1:
        return None

    lines = hunk.get("lines") or []
    new_start = hunk.get("new_start")
    new_len   = hunk.get("new_len")
    new_end   = (new_start + max(0, new_len) - 1) if isinstance(new_start, int) and isinstance(new_len, int) else None

    # Build +/- list (what the model saw in the prompt)
    plus_minus = [(i, e) for i, e in enumerate(lines) if e.get("tag") in {"+", "-"}]

    def new_if_add(e: dict) -> int | None:
        return e.get("new") if e.get("tag") == "+" and isinstance(e.get("new"), int) else None

    # A) index over +/- list
    if plus_minus and approx_line <= len(plus_minus):
        _, e = plus_minus[approx_line - 1]
        nl = new_if_add(e)
        if nl is not None:
            return nl
        # search forward then backward for a '+' anchor
        for _, ee in plus_minus[approx_line:]:
            nl = new_if_add(ee)
            if nl is not None:
                return nl
        for _, ee in reversed(plus_minus[:approx_line - 1]):
            nl = new_if_add(ee)
            if nl is not None:
                return nl

    # B) absolute new-file line within hunk
    if isinstance(new_start, int) and isinstance(new_end, int):
        if new_start <= approx_line <= new_end:
            return approx_line

    # C) offset from new_start
    if isinstance(new_start, int):
        candidate = new_start + (approx_line - 1)
        if isinstance(new_end, int) and new_start <= candidate <= new_end:
            return candidate

    # D) fallback to any '+' line in the hunk
    for e in lines:
        nl = new_if_add(e)
        if nl is not None:
            return nl

    return None




def process_diff_line(
    line_obj: Dict[str, Any],
    repo: str,
    pr: int,
    file_path: str,
    model_name: str,
    shot: str,
    hunk_idx: int,
    hunk: Dict[str, Any],
    prompt_used: str,
) -> Dict[str, Any]:
    line_val = line_obj.get("line")
    comment = line_obj.get("comment")
    ctype = (line_obj.get("type") or "").upper()

    return {
        "repo": repo,
        "pr_id": pr,
        "file_path": file_path,
        "hunk_index": hunk_idx,
        "hunk_header": hunk.get("header"),
        "new_start": hunk.get("new_start"),
        "new_len": hunk.get("new_len"),
        "old_start": hunk.get("old_start"),
        "old_len": hunk.get("old_len"),
        "model": model_name,
        "shot": shot,
        "generated_line": line_val,
        "generated_line_abs_new": absolute_new_line(hunk, line_val),
        "generated_type": ctype,
        "generated_comment": comment,
        "prompt_chars": len(prompt_used),
        "ts": int(time.time()),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate review comments from diff_extractor output using Ollama SLMs.")
    ap.add_argument("repo", type=str, help="Repo full name, e.g. localstack/localstack")
    ap.add_argument("pr", type=int, help="Pull Request number")
    ap.add_argument("diff_jsonl", type=str, help="Path to diff_extractor JSONL output")
    ap.add_argument("model", type=str, choices=list(MODEL_MAP.keys()), help="SLM key: phi|mistral|gemma")
    ap.add_argument("shot", type=int, choices=[0, 1], help="0 = zero-shot, 1 = few-shot")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT_PATH, help="Output JSONL path")
    ap.add_argument("--max", type=int, default=0, help="Max hunks to process (0 = all)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Ollama request timeout (s)")
    args = ap.parse_args()

    model_name = MODEL_MAP[args.model]
    shot_name = "few" if args.shot == 1 else "zero"
    prompt_path = PROMPT_FEW_PATH if args.shot == 1 else PROMPT_ZERO_PATH
    prompt_template = load_prompt_text(prompt_path)

    diff_path = Path(args.diff_jsonl)
    if not diff_path.exists():
        print(f"[ERROR] Diff file not found: {diff_path}")
        sys.exit(1)

    print(f"[INFO] Model: {model_name} | Shot: {shot_name}")
    print(f"[INFO] Repo: {args.repo} | PR: {args.pr}")
    print(f"[INFO] Input: {diff_path}")
    print(f"[INFO] Output: {args.out}\n")

    processed = written = 0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as fout, open(diff_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            file_path = rec.get("path")
            hunks = rec.get("hunks") or []
            if not file_path or not hunks:
                continue

            for h_idx, hunk in enumerate(hunks):
                if args.max and processed >= args.max:
                    break

                context = build_context_from_hunk(hunk)
                diff_txt = build_diff_from_hunk(hunk)
                prompt = fill_prompt(prompt_template, context, diff_txt)

                print(f"[HUNK] {file_path} | hunk {h_idx} | chars(context)={len(context)} chars(diff)={len(diff_txt)}")

                try:
                    raw = ask_ollama(prompt, model_name, timeout_s=args.timeout)
                except Exception as e:
                    fail_row = {
                        "repo": args.repo,
                        "pr_id": args.pr,
                        "file_path": file_path,
                        "hunk_index": h_idx,
                        "model": model_name,
                        "shot": shot_name,
                        "parse_fail": True,
                        "error": str(e),
                        "raw": None,
                        "ts": int(time.time()),
                    }
                    fout.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
                    written += 1
                    processed += 1
                    continue

                parsed, parse_fail = parse_model_json_strict(raw)
                if parse_fail:
                    fail_row = {
                        "repo": args.repo,
                        "pr_id": args.pr,
                        "file_path": file_path,
                        "hunk_index": h_idx,
                        "model": model_name,
                        "shot": shot_name,
                        "parse_fail": True,
                        "raw": raw,
                        "ts": int(time.time()),
                    }
                    fout.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
                    written += 1
                else:
                    # --- Filter multiple comments per hunk ---
                    filtered = [p for p in parsed if p.get("comment") and "looks good" not in p["comment"].lower()]
                    if not filtered:
                        # If all were trivial, just keep the first
                        filtered = parsed[:1]
                    else:
                        # Keep the longest informative one
                        filtered = [max(filtered, key=lambda x: len(x.get("comment", "")))]

                    # --- Write only one final record ---
                    item = filtered[0]
                    row = process_diff_line(
                        item,
                        repo=args.repo,
                        pr=args.pr,
                        file_path=file_path,
                        model_name=model_name,
                        shot=shot_name,
                        hunk_idx=h_idx,
                        hunk=hunk,
                        prompt_used=prompt,
                    )
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1


                processed += 1
            if args.max and processed >= args.max:
                break

    print(f"\n[DONE] hunks processed: {processed} | rows written: {written} | out: {args.out}")


if __name__ == "__main__":
    main()
