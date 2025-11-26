from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from .common import gh_get, load_github_token, API_BASE, diff_path, append_jsonl, log_info

import re

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# ----------------------------- helpers --------------------------------------

def list_pr_files(owner: str, repo: str, pr_number: int, token: Optional[str]):
    """Iterate all files in a PR (handles pagination)."""
    page = 1
    per_page = 100
    while True:
        r = gh_get(
            f"{API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}/files",
            token,
            params={"page": page, "per_page": per_page},
        )
        items = r.json()
        if not items:
            break
        for it in items:
            yield it
        if len(items) < per_page:
            break
        page += 1

def split_hunks(patch: str):
    """Split a full patch into @@ ... @@ blocks."""
    if not patch:
        return []
    chunks = []
    cur = []
    for line in patch.splitlines():
        if line.startswith("@@") and cur:
            chunks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def parse_diff_hunk(diff_text: str) -> Optional[Dict[str, Any]]:
    """Parse one @@ hunk into structured form."""
    if not diff_text:
        return None
    lines = diff_text.splitlines()
    header_line = next((l for l in lines if l.startswith("@@")), None)
    if not header_line:
        return None

    m = HUNK_RE.match(header_line.strip())
    if not m:
        return None

    old_start = int(m.group(1))
    old_len = int(m.group(2) or "1")
    new_start = int(m.group(3))
    new_len = int(m.group(4) or "1")

    out_lines = []
    old_no, new_no = old_start, new_start
    for raw in lines[1:]:
        if not raw:
            out_lines.append({"tag": " ", "text": "", "old": old_no, "new": new_no})
            old_no += 1
            new_no += 1
            continue
        tag = raw[0]
        text = raw[1:] if tag in "+- " else raw
        if tag == "+":
            out_lines.append({"tag": "+", "text": text, "old": None, "new": new_no})
            new_no += 1
        elif tag == "-":
            out_lines.append({"tag": "-", "text": text, "old": old_no, "new": None})
            old_no += 1
        else:
            out_lines.append({"tag": " ", "text": text, "old": old_no, "new": new_no})
            old_no += 1
            new_no += 1

    return {
        "header": header_line,
        "old_start": old_start,
        "old_len": old_len,
        "new_start": new_start,
        "new_len": new_len,
        "lines": out_lines,
    }

def build_position_table(patch: str) -> Dict[int, int]:
    """Map new line numbers → unified diff positions."""
    if not patch:
        return {}
    pos_table = {}
    pos = 0
    for hunk_text in split_hunks(patch):
        parsed = parse_diff_hunk(hunk_text)
        if not parsed:
            continue
        for ln in parsed["lines"]:
            pos += 1
            if ln["tag"] in ("+", " "):
                new_no = ln["new"]
                if new_no is not None:
                    pos_table[new_no] = pos
    return pos_table

# ----------------------------- main extractor --------------------------------------

def extract_pr_diffs(owner_repo: str, pr_number: int) -> Path:
    """Fetch and save all diffs for a PR."""
    token = load_github_token()
    owner, repo = owner_repo.split("/", 1)
    out_path = diff_path(owner_repo, pr_number)
    out_path.unlink(missing_ok=True)
    for f in list_pr_files(owner, repo, pr_number, token):
        patch = f.get("patch")
        hunks = [parse_diff_hunk(h) for h in split_hunks(patch or "") if h]
        pos_table = build_position_table(patch or "")
        rec = {
            "repo": owner_repo,
            "pr_id": pr_number,
            "path": f.get("filename"),
            "status": f.get("status"),
            "previous_filename": f.get("previous_filename"),
            "patch": patch,
            "hunks": hunks,
            "position_table": pos_table,
        }
        append_jsonl(out_path, rec)
    log_info(f"Diffs saved to {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m lite_reviewer.extractor <owner/repo> <pr>")
        sys.exit(1)
    extract_pr_diffs(sys.argv[1], int(sys.argv[2]))
