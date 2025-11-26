import os
import sys
import json
import requests
from pathlib import Path

GITHUB_API = "https://api.github.com"

# Default paths (change if you want)
DEFAULT_REVIEWS_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\data\generated_reviews.jsonl"

def main():
    if len(sys.argv) < 3:
        print("Usage: python post_comments_github.py <repo> <pr_id> [reviews_path]")
        sys.exit(1)

    repo = sys.argv[1]                 # e.g. "localstack/localstack"
    pr_id = int(sys.argv[2])           # e.g. 9866
    reviews_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_REVIEWS_PATH

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("[ERROR] Please set GITHUB_TOKEN environment variable.")
        sys.exit(1)

    p = Path(reviews_path)
    if not p.exists():
        print(f"[ERROR] Reviews file not found: {reviews_path}")
        sys.exit(1)

    # Load rows (each row corresponds to a hunk prediction from your generator)
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("parse_fail"):
                continue
            # must have these to span a hunk
            if not r.get("file_path") or not r.get("hunk_header"):
                continue
            rows.append(r)

    if not rows:
        print("[INFO] Nothing to post.")
        return

    # Build multi-line review comments; one per hunk row
    comments_payload = []
    for r in rows:
        path = r["file_path"]
        body = r.get("generated_comment") or "Looks good to me."
        # Prefer additions; otherwise deletions
        new_start = r.get("new_start")
        new_len   = r.get("new_len")
        old_start = r.get("old_start")
        old_len   = r.get("old_len")

        # Normalize ints
        new_start = new_start if isinstance(new_start, int) else None
        new_len   = new_len   if isinstance(new_len, int)   else None
        old_start = old_start if isinstance(old_start, int) else None
        old_len   = old_len   if isinstance(old_len, int)   else None

        # Decide side and range
        if new_start is not None and new_len is not None and new_len > 0:
            side = "RIGHT"
            start_line = new_start
            end_line   = new_start + new_len - 1
        elif old_start is not None and old_len is not None and old_len > 0:
            side = "LEFT"
            start_line = old_start
            end_line   = old_start + old_len - 1
        else:
            # Fallback: skip if we can’t derive a sensible span
            print(f"[SKIP] No valid hunk span for {path}")
            continue

        # GitHub review API: if the span is 1 line, use single-line form.
        if start_line == end_line:
            comments_payload.append({
                "path": path,
                "line": start_line,
                "side": side,
                "body": body
            })
        else:
            comments_payload.append({
                "path": path,
                "start_line": start_line,
                "start_side": side,
                "line": end_line,
                "side": side,
                "body": body
            })

    if not comments_payload:
        print("[INFO] No comments to post after filtering.")
        return

    # Create a single review with all comments (no commit_id/positions needed)
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_id}/reviews"
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}",
               "Accept": "application/vnd.github+json"}
    payload = {
        "event": "COMMENT",
        "comments": comments_payload
        # optional: "body": "Automated review by LiteReviewer"
    }

    print(f"[INFO] Posting {len(comments_payload)} hunk-wide comments as one review…")
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code not in (200, 201):
        print(f"[ERROR] Failed to create review: {r.status_code} {r.text}")
        sys.exit(1)
    print(f"[OK] Review created with {len(comments_payload)} comments.")

if __name__ == "__main__":
    main()
