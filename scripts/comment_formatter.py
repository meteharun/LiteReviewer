import json
import argparse
from pathlib import Path
from typing import List, Dict

# Input and output paths
DEFAULT_IN_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\data\generated_reviews.jsonl"
DEFAULT_OUT_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\data\formatted_comments.json"


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def format_comments(records: List[Dict]) -> List[Dict]:
    formatted = []
    for rec in records:
        if rec.get("parse_fail"):
            continue
        file_path = rec.get("file_path")
        line = rec.get("generated_line_abs_new")
        comment = rec.get("generated_comment")
        if not (file_path and line and comment):
            continue
        formatted.append({
            "path": file_path,
            "line": int(line),
            "body": comment.strip(),
            "side": "RIGHT"
        })
    return formatted


def main():
    ap = argparse.ArgumentParser(description="Format generated review comments for CI integration.")
    ap.add_argument("--infile", type=str, default=DEFAULT_IN_PATH, help="Input JSONL file (from generator).")
    ap.add_argument("--outfile", type=str, default=DEFAULT_OUT_PATH, help="Output JSON file (CI-ready).")
    args = ap.parse_args()

    if not Path(args.infile).exists():
        print(f"[ERROR] Input file not found: {args.infile}")
        return

    data = load_jsonl(args.infile)
    formatted = format_comments(data)

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Formatted {len(formatted)} comments → {args.outfile}")


if __name__ == "__main__":
    main()
