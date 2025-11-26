# lite_reviewer/poster.py
from __future__ import annotations
import json, sys
from typing import Dict, Any, Optional, List
from .common import (
    diff_path, reviews_path, load_github_token, gh_get, gh_post,
    API_BASE, log_info, log_err, resolve_model
)

BOT_MARK = "<!-- LiteReviewer -->"

def get_latest_commit_sha(repo: str, pr: int, token: str) -> str:
    r = gh_get(f"{API_BASE}/repos/{repo}/pulls/{pr}", token)
    r.raise_for_status()
    return r.json()["head"]["sha"]

# ---------- Load hunk spans (unified diff positions) from diff JSONL ----------

def _load_hunk_spans(diff_jsonl_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns { path: [ {start_pos, end_pos, new_start, new_len, old_start, old_len}, ... ] }
    start_pos/end_pos are 1-based unified *position indices* within that file’s patch.
    """
    spans: Dict[str, List[Dict[str, Any]]] = {}

    try:
        with open(diff_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                path = rec.get("path")
                hunks = rec.get("hunks") or []
                if not path or not hunks:
                    continue

                pos = 0  # position counter across this file’s patch
                file_spans: List[Dict[str, Any]] = []
                for h in hunks:
                    lines = h.get("lines") or []
                    if not lines:
                        continue
                    start_pos = pos + 1           # first body line after header
                    pos += len(lines)             # each body line increments position
                    end_pos = pos                 # last body line

                    file_spans.append({
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "new_start": h.get("new_start"),
                        "new_len":   h.get("new_len"),
                        "old_start": h.get("old_start"),
                        "old_len":   h.get("old_len"),
                    })

                if file_spans:
                    spans[path] = file_spans
    except FileNotFoundError:
        # Caller handles error
        pass

    return spans

def _span_from_row_basic(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Use hunk bounds present in the row (preferred).
    Returns {path, start_line, end_line, side} or None.
    """
    path = row.get("file_path")
    if not path:
        return None

    new_start = row.get("new_start")
    new_len   = row.get("new_len")
    old_start = row.get("old_start")
    old_len   = row.get("old_len")

    new_start = new_start if isinstance(new_start, int) else None
    new_len   = new_len   if isinstance(new_len, int)   else None
    old_start = old_start if isinstance(old_start, int) else None
    old_len   = old_len   if isinstance(old_len, int)   else None

    if new_start is not None and new_len and new_len > 0:
        return {
            "path": path,
            "start_line": new_start,
            "end_line": new_start + new_len - 1,
            "side": "RIGHT",
        }
    if old_start is not None and old_len and old_len > 0:
        return {
            "path": path,
            "start_line": old_start,
            "end_line": old_start + old_len - 1,
            "side": "LEFT",
        }
    return None

def _span_from_position(path: str, chosen_position: int,
                        hunk_spans: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Map a chosen unified 'position' to its hunk in this file, then
    derive a span from the hunk’s new/old bounds.
    """
    if not isinstance(chosen_position, int):
        return None
    file_spans = hunk_spans.get(path) or []
    for hs in file_spans:
        if hs["start_pos"] <= chosen_position <= hs["end_pos"]:
            n_s, n_l = hs.get("new_start"), hs.get("new_len")
            o_s, o_l = hs.get("old_start"), hs.get("old_len")
            if isinstance(n_s, int) and isinstance(n_l, int) and n_l > 0:
                return {
                    "path": path,
                    "start_line": n_s,
                    "end_line": n_s + n_l - 1,
                    "side": "RIGHT",
                }
            if isinstance(o_s, int) and isinstance(o_l, int) and o_l > 0:
                return {
                    "path": path,
                    "start_line": o_s,
                    "end_line": o_s + o_l - 1,
                    "side": "LEFT",
                }
            # If both are unusable, we can’t form a span
            return None
    return None

# ---------- Posting ----------

def _post_hunk_review(repo: str, pr: int, token: str,
                      span: Dict[str, Any], body: str, dry_run: bool = False):
    """
    Create a single review with one hunk-wide comment using start_line/end_line.
    """
    if span["start_line"] == span["end_line"]:
        comment_item = {
            "path": span["path"],
            "line": span["end_line"],
            "side": span["side"],
            "body": f"{BOT_MARK}\n{body}",
        }
    else:
        comment_item = {
            "path": span["path"],
            "start_line": span["start_line"],
            "start_side": span["side"],
            "line": span["end_line"],
            "side": span["side"],
            "body": f"{BOT_MARK}\n{body}",
        }

    if dry_run:
        print("[DRY-RUN] Would create review with comment:", json.dumps(comment_item, ensure_ascii=False))
        return

    payload = {
        "event": "COMMENT",
        "comments": [comment_item],
    }
    r = gh_post(f"{API_BASE}/repos/{repo}/pulls/{pr}/reviews", token, payload)
    if r.status_code in (200, 201):
        log_info(f"Posted review on {span['path']} [{span['side']}] {span['start_line']}–{span['end_line']}")
    else:
        log_err(f"Failed to create review: {r.status_code} {r.text}")

def post_from_reviews(owner_repo: str, pr: int, model_key: str, shot: str, dry_run: bool = False):
    token = load_github_token()
    if not token:
        log_err("GITHUB_TOKEN not set or missing in github_token.txt")
        sys.exit(1)

    model_id = resolve_model(model_key)
    diff_file = diff_path(owner_repo, pr)
    rev_file  = reviews_path(owner_repo, pr, model_id, shot)

    log_info(f"Posting comments from {rev_file}")

    # Load diff hunk spans (for mapping chosen_position → hunk)
    hunk_spans = _load_hunk_spans(diff_file)

    try:
        with open(rev_file, "r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        log_err(f"Reviews file not found: {rev_file}")
        sys.exit(1)

    posted = 0
    for r in rows:
        if r.get("parse_fail"):
            continue
        body = (r.get("generated_comment") or "").strip()
        if not body:
            continue
        path = r.get("file_path")
        if not path:
            continue

        # 1) Try span directly from row hunk bounds
        span = _span_from_row_basic(r)
        # 2) Fallback: map chosen_position → hunk from diff file
        if not span:
            chosen_pos = r.get("chosen_position")
            span = _span_from_position(path, chosen_pos, hunk_spans)

        if not span:
            # Nothing we can do for this row
            continue

        _post_hunk_review(owner_repo, pr, token, span, body, dry_run=dry_run)
        posted += 1

    if posted == 0:
        log_info("No comments to post after filtering.")
    else:
        log_info(f"Posted {posted} comment(s).")
