import sys
import json
import re
import requests
from pathlib import Path

API_BASE = "https://api.github.com"
TOKEN_PATH = "github_token.txt"

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def load_token() -> str | None:
    p = Path(TOKEN_PATH)
    if p.exists():
        return p.read_text(encoding="utf-8").strip() or None
    return None

def gh_get(url: str, token: str | None, params=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "LiteReviewer/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r

def list_pr_files(owner: str, repo: str, pr_number: int, token: str | None):
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

def parse_diff_hunk(diff_text: str):
    """
    Parse a single unified diff hunk into a structure with line-by-line tags and
    old/new line numbers. Supports leading ' ', '+', '-' lines.
    Returns dict with old/new ranges and a 'lines' list.
    """
    if not diff_text:
        return None
    lines = diff_text.splitlines()
    # Find header
    header_idx = None
    header_line = None
    for i, l in enumerate(lines):
        if l.startswith("@@"):
            header_idx = i
            header_line = l
            break
    if header_idx is None:
        return None

    m = HUNK_RE.match(header_line.strip())
    if not m:
        return None

    old_start = int(m.group(1))
    old_len = int(m.group(2) or "1")
    new_start = int(m.group(3))
    new_len = int(m.group(4) or "1")

    out_lines = []
    old_no = old_start
    new_no = new_start

    for raw in lines[header_idx + 1:]:
        if not raw:
            # treat as context (empty line), but unified diff usually has a leading ' ' on contexts
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
            # context
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

def split_hunks(patch: str):
    """
    Split a full file patch into individual hunks (each starting with @@ ... @@).
    Returns list of hunk strings (including their @@ header).
    """
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

def build_unified_position_table(patch: str):
    """
    Build a mapping of new-file line -> unified diff position index (1-based positions as GitHub expects).
    The 'position' is the index of the line within the *file's* patch across all hunks.
    We count every line in the patch after each hunk header as a position step.
    """
    if not patch:
        return {}

    pos_table = {}  # new_line_no -> unified position
    pos = 0
    for hunk_text in split_hunks(patch):
        lines = hunk_text.splitlines()
        # first line is header; positions start after header
        for i, raw in enumerate(lines):
            if i == 0:
                continue
            pos += 1
            tag = raw[0] if raw else " "
            if tag == "+" or tag == " ":
                # compute the new line number while walking
                # we need to simulate like in parse_diff_hunk
                # simpler: re-parse hunk and then re-count with alignment
                pass

    # The above naive approach isn't enough to know new line per position.
    # Instead, properly parse each hunk and advance a running 'pos' counter in sync.

    pos_table.clear()
    pos = 0
    for hunk_text in split_hunks(patch):
        parsed = parse_diff_hunk(hunk_text)
        if not parsed:
            continue
        # iterate hunk body; each body line increments position
        for ln in parsed["lines"]:
            pos += 1
            if ln["tag"] in ("+", " "):
                new_no = ln["new"]
                if new_no is not None and new_no not in pos_table:
                    pos_table[new_no] = pos
    return pos_table

def extract_pr_diffs(owner: str, repo: str, pr_number: int, token: str | None):
    """
    Yield per-file diff info:
    {
      "path": str,
      "status": "modified|added|removed|renamed|...",
      "previous_filename": str|None,
      "patch": str|None,
      "hunks": [ parsed_hunk, ... ],
      "position_table": { new_line -> position }
    }
    """
    for f in list_pr_files(owner, repo, pr_number, token):
        path = f.get("filename")
        status = f.get("status")
        prev = f.get("previous_filename")
        patch = f.get("patch")  # None for binary or move-only

        if not patch:
            yield {
                "path": path,
                "status": status,
                "previous_filename": prev,
                "patch": None,
                "hunks": [],
                "position_table": {},
            }
            continue

        hunks_raw = split_hunks(patch)
        hunks_parsed = [h for h in (parse_diff_hunk(hh) for hh in hunks_raw) if h]
        pos_table = build_unified_position_table(patch)

        yield {
            "path": path,
            "status": status,
            "previous_filename": prev,
            "patch": patch,
            "hunks": hunks_parsed,
            "position_table": pos_table,
        }

def main():
    if len(sys.argv) != 3:
        print("Usage: python diff_extractor.py <owner/repo> <pr_number>")
        sys.exit(1)

    full = sys.argv[1]
    pr_number = int(sys.argv[2])
    if "/" not in full:
        print("Repo must be in the form owner/repo")
        sys.exit(1)
    owner, repo = full.split("/", 1)

    token = load_token()

    for file_info in extract_pr_diffs(owner, repo, pr_number, token):
        print(json.dumps(file_info, ensure_ascii=False))

if __name__ == "__main__":
    main()
