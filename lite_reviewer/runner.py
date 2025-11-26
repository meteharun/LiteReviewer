from __future__ import annotations
import argparse
from .extractor import extract_pr_diffs
from .generator import generate_reviews
from .poster import post_from_reviews
from .common import log_info

def main():
    ap = argparse.ArgumentParser(description="Run full LiteReviewer pipeline.")
    ap.add_argument("repo", help="owner/repo")
    ap.add_argument("pr", type=int)
    ap.add_argument("--model", choices=["phi", "mistral", "gemma"], default="phi")
    ap.add_argument("--shot", choices=["zero", "few"], default="zero")
    ap.add_argument("--max-hunks", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_info(f"Extracting diffs for {args.repo}#{args.pr}")
    diff_file = extract_pr_diffs(args.repo, args.pr)

    log_info("Generating reviews...")
    reviews_file = generate_reviews(args.repo, args.pr, args.model, args.shot, args.max_hunks)

    log_info("Posting comments to GitHub...")
    post_from_reviews(args.repo, args.pr, args.model, args.shot, dry_run=args.dry_run)

    log_info(f"Done. Results: {reviews_file}")

if __name__ == "__main__":
    main()
