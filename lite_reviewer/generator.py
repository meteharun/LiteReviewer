from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    diff_path, reviews_path, load_prompt, fill_prompt,
    ollama_generate, parse_model_json, choose_position_for_hunk,
    append_jsonl, log_info, log_warn, log_err, resolve_model, now_ts
)

# Clamp sizes so small models on CPU don’t drift/hallucinate
MAX_CONTEXT_LINES = 80           # neutral lines kept from hunk
MAX_DIFF_BODY_LINES = 160        # +/- lines kept from hunk body
SHOW_PROMPT_PREFIX = 400         # how much of prompt to echo for debugging

def _take_head_tail(items: List[str], limit: int) -> List[str]:
    if len(items) <= limit:
        return items
    half = limit // 2
    return items[:half] + ["..."] + items[-(limit - half):]

def build_context(hunk: Dict[str, Any]) -> str:
    neutral = [ln.get("text", "") for ln in hunk.get("lines", []) if ln.get("tag") == " "]
    neutral = _take_head_tail(neutral, MAX_CONTEXT_LINES)
    return "\n".join(neutral)

def build_diff_text(hunk: Dict[str, Any]) -> str:
    header = hunk.get("header", "")
    removed = []
    added = []
    for ln in hunk.get("lines", []):
        if ln["tag"] == "-":
            removed.append(ln["text"])
        elif ln["tag"] == "+":
            added.append(ln["text"])
    parts = [header, "Removed:"]
    parts += ["  " + s for s in removed] or ["  (none)"]
    parts += ["Added:"]
    parts += ["  " + s for s in added] or ["  (none)"]
    return "\n".join(parts)


def _first_valid_comment(items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for it in items:
        c = (it.get("comment") or "").strip()
        if c:
            # normalize type (optional for posting)
            t = (it.get("type") or "other").upper()
            return {"comment": c, "type": t, "line": it.get("line")}
    return None

def generate_reviews(owner_repo: str, pr_number: int, model_key: str, shot: str, max_hunks: int = 0):
    diff_file = diff_path(owner_repo, pr_number)
    model = resolve_model(model_key)
    prompt_template = load_prompt(f"{shot}_shot.json")
    out_file = reviews_path(owner_repo, pr_number, model, shot)

    with open(diff_file, "r", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    processed = 0
    for rec in recs:
        file_path = rec.get("path")
        if not file_path:
            continue
        pos_table = rec.get("position_table", {}) or {}
        hunks = rec.get("hunks", []) or []

        for h_idx, hunk in enumerate(hunks):
            if max_hunks and processed >= max_hunks:
                break

            context = build_context(hunk)
            diff_txt = build_diff_text(hunk)
            prompt = fill_prompt(prompt_template, context, diff_txt)

            # Helpful debug to see if prompt exploded
            if processed == 0:
                log_info(f"Prompt preview ({shot}/{model}): {prompt[:SHOW_PROMPT_PREFIX].replace(chr(10),' ')} ... [len={len(prompt)}]")

            try:
                raw = ollama_generate(
                    prompt, model,
                    # extra safety for few-shot
                    num_ctx=2048 if shot == "few" else 1536,
                    temperature=0.0
                )
                parsed = parse_model_json(raw)

                # Fallback once with minimal zero-shot if few-shot drifts
                if not parsed and shot == "few":
                    log_warn("Few-shot parse failed; retrying with zero-shot fallback on this hunk.")
                    zs_tmpl = load_prompt("zero_shot.json")
                    zs_prompt = fill_prompt(zs_tmpl, context, diff_txt)
                    raw = ollama_generate(zs_prompt, model, num_ctx=1536, temperature=0.0)
                    parsed = parse_model_json(raw)

                if not parsed:
                    append_jsonl(out_file, {
                        "repo": owner_repo, "pr_id": pr_number,
                        "file_path": file_path, "hunk_index": h_idx,
                        "model": model, "shot": shot,
                        "parse_fail": True, "raw": raw, "ts": now_ts()
                    })
                    continue

                picked = _first_valid_comment(parsed)
                if not picked:
                    append_jsonl(out_file, {
                        "repo": owner_repo, "pr_id": pr_number,
                        "file_path": file_path, "hunk_index": h_idx,
                        "model": model, "shot": shot,
                        "parse_fail": True, "raw": raw, "ts": now_ts()
                    })
                    continue

                comment = picked["comment"]
                ctype = picked["type"]
                pos = choose_position_for_hunk(hunk, pos_table)

                append_jsonl(out_file, {
                    "repo": owner_repo, "pr_id": pr_number,
                    "file_path": file_path, "hunk_index": h_idx,
                    "model": model, "shot": shot,
                    "generated_type": ctype, "generated_comment": comment,
                    "chosen_position": pos, "ts": now_ts()
                })
                processed += 1

            except Exception as e:
                append_jsonl(out_file, {
                    "repo": owner_repo, "pr_id": pr_number,
                    "file_path": file_path, "hunk_index": h_idx,
                    "model": model, "shot": shot,
                    "parse_fail": True, "error": str(e), "ts": now_ts()
                })

    log_info(f"Reviews saved to {out_file}")
    return out_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python -m lite_reviewer.generator <repo> <pr> <model_key> <shot>")
        sys.exit(1)
    generate_reviews(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
