import json
import os
import re
from github import Github, GithubException

# ================== CONFIG ==================
TOKEN_PATH = "github_token.txt"

REPO_LIST_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\dataset\top_python_repos.jsonl"
OUTPUT_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\dataset\pr_review_samples.jsonl"

MAX_EXAMPLES_PER_REPO = 100
MAX_PRS_TO_SCAN = 300
CONTEXT_RADIUS = 5
MAX_DEF_BLOCK_LINES = 300
REPO_LIMIT = 50

# rule parameters
PROBE_PRS = 10       # check first 10 PRs
PROBE_MIN_EXAMPLES = 2
# ============================================


def load_token(token_path):
    with open(token_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_repo_list(jsonl_path):
    repos = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            repos.append(json.loads(line))
    return repos


def is_bot(user_obj):
    if user_obj is None:
        return True
    if getattr(user_obj, "type", None) == "Bot":
        return True
    login = getattr(user_obj, "login", "") or ""
    if login.endswith("[bot]"):
        return True
    return False


def get_file_content_at_commit(repo, file_path, sha):
    if sha is None:
        return None
    try:
        file_content = repo.get_contents(file_path, ref=sha)
    except GithubException:
        return None
    text = file_content.decoded_content.decode("utf-8", errors="replace")
    return text.splitlines()


def clamp_line(num, lo, hi):
    return max(lo, min(num, hi))


########################
# diff hunk parsing
########################

HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def parse_diff_hunk(diff_text):
    if not diff_text:
        return None
    lines = diff_text.splitlines()
    if not lines:
        return None

    header_idx = None
    header_line = None
    for i, l in enumerate(lines):
        if l.strip().startswith("@@"):
            header_idx = i
            header_line = l.strip()
            break
    if header_idx is None:
        return None

    m = HEADER_RE.match(header_line)
    if not m:
        return None

    old_start = int(m.group(1))
    old_len = int(m.group(2) or "1")
    new_start = int(m.group(3))
    new_len = int(m.group(4) or "1")

    hunk_lines = []
    old_line_no = old_start
    new_line_no = new_start

    for l in lines[header_idx + 1:]:
        if l.startswith('+'):
            hunk_lines.append(('+', l[1:], None, new_line_no))
            new_line_no += 1
        elif l.startswith('-'):
            hunk_lines.append(('-', l[1:], old_line_no, None))
            old_line_no += 1
        else:
            text = l[1:] if l.startswith(' ') else l
            hunk_lines.append((' ', text, old_line_no, new_line_no))
            old_line_no += 1
            new_line_no += 1

    return {
        "header": header_line,   # <- keep header for grouping
        "old_start": old_start,
        "old_len": old_len,
        "new_start": new_start,
        "new_len": new_len,
        "lines": hunk_lines,
    }


def get_old_span_from_hunk(hunk):
    if hunk is None:
        return None, None

    old_nos = [
        old_no for (tag, text, old_no, new_no) in hunk["lines"]
        if tag in (' ', '-') and old_no is not None
    ]
    if old_nos:
        return min(old_nos), max(old_nos)

    start = hunk["old_start"]
    end = hunk["old_start"] + max(hunk["old_len"], 1) - 1
    return start, end


########################
# context extraction
########################

def extract_block_definition(old_lines, anchor_line):
    if not old_lines or anchor_line is None:
        return None

    n = len(old_lines)
    anchor_line = clamp_line(anchor_line, 1, n)
    idx = anchor_line - 1

    def_idx = None
    def_indent = None
    i = idx
    while i >= 0:
        li = old_lines[i]
        stripped = li.lstrip()
        if stripped.startswith("def "):
            def_idx = i
            def_indent = len(li) - len(stripped)
            break
        i -= 1

    class_idx = None
    class_indent = None
    if def_idx is None:
        j = idx
        while j >= 0:
            lj = old_lines[j]
            stripped = lj.lstrip()
            if stripped.startswith("class "):
                class_idx = j
                class_indent = len(lj) - len(stripped)
                break
            j -= 1

    if def_idx is not None:
        anchor_start = def_idx
        anchor_indent = def_indent
    elif class_idx is not None:
        anchor_start = class_idx
        anchor_indent = class_indent
    else:
        return None

    end_idx = anchor_start
    k = anchor_start + 1
    while k < n:
        lk = old_lines[k]
        if lk.strip() == "":
            end_idx = k
            k += 1
            continue
        cur_indent = len(lk) - len(lk.lstrip())
        if cur_indent <= anchor_indent and not lk.lstrip().startswith("@"):
            break
        end_idx = k
        k += 1

    block = old_lines[anchor_start:end_idx + 1]
    if len(block) > MAX_DEF_BLOCK_LINES:
        return None
    return "\n".join(block)


def extract_window_snippet(old_lines, start_line, end_line, radius):
    if not old_lines or start_line is None or end_line is None:
        return None

    n = len(old_lines)
    s = clamp_line(start_line - radius, 1, n)
    e = clamp_line(end_line + radius, 1, n)
    return "\n".join(old_lines[s - 1:e])


########################
# main per-PR scraping
########################

def scrape_examples_from_single_pr(repo, pr, max_needed):
    out = []
    pr_author = pr.user.login if pr.user else None
    pr_base_sha = getattr(pr.base, "sha", None)

    try:
        review_comments = list(pr.get_review_comments())
    except GithubException:
        return out

    # 1) apply all per-comment filters FIRST
    filtered = []
    for c in review_comments:
        if getattr(c, "in_reply_to_id", None) is not None:
            continue
        if is_bot(c.user):
            continue
        comment_author = c.user.login if c.user else None
        if comment_author == pr_author:
            continue
        file_path = getattr(c, "path", None)
        if not file_path or not file_path.endswith(".py"):
            continue
        diff_hunk = getattr(c, "diff_hunk", None)
        if not diff_hunk:
            continue
        body_text = (c.body or "").strip()
        if not body_text:
            continue
        filtered.append(c)

    # 2) group AFTER filtering by (file_path, hunk header)
    groups = {}
    for c in filtered:
        file_path = getattr(c, "path", None)
        diff_hunk = getattr(c, "diff_hunk", None)
        # extract header (@@ -a,b +c,d @@ ...)
        header_line = None
        for ln in diff_hunk.splitlines():
            if ln.strip().startswith("@@"):
                header_line = ln.strip()
                break
        key = (file_path, header_line)  # <- use header, not full hunk body
        groups.setdefault(key, []).append(c)

    # 3) keep only hunks that have exactly one comment
    singletons = {key for key, cs in groups.items() if len(cs) == 1}

    # 4) emit examples only for singleton hunks
    for c in filtered:
        if len(out) >= max_needed:
            break

        file_path = getattr(c, "path", None)
        diff_hunk = getattr(c, "diff_hunk", None)

        # compute header again to check membership
        header_line = None
        for ln in diff_hunk.splitlines():
            if ln.strip().startswith("@@"):
                header_line = ln.strip()
                break
        key = (file_path, header_line)
        if key not in singletons:
            continue

        comment_author = c.user.login if c.user else None
        body_text = (c.body or "").strip()

        comment_id = getattr(c, "id", None)
        comment_url = (
            f"https://github.com/{repo.full_name}/pull/{pr.number}#discussion_r{comment_id}"
            if comment_id else None
        )

        hunk = parse_diff_hunk(diff_hunk)
        old_span_start, old_span_end = get_old_span_from_hunk(hunk)
        old_file_lines = get_file_content_at_commit(repo, file_path, pr_base_sha)

        context_text = None
        if old_file_lines and old_span_start and old_span_end:
            anchor_line = (old_span_start + old_span_end) // 2
            block_snippet = extract_block_definition(old_file_lines, anchor_line)
            if block_snippet:
                context_text = block_snippet
            else:
                context_text = extract_window_snippet(
                    old_file_lines, old_span_start, old_span_end, CONTEXT_RADIUS
                )

        example = {
            "repo": repo.full_name,
            "pr_id": pr.number,
            "file_path": file_path,
            "diff_hunk": diff_hunk,
            "context": context_text,
            "comment": body_text,
            "comment_author": comment_author,
            "comment_url": comment_url,
        }
        out.append(example)

    return out


########################
# repo loop with probe rule
########################

def collect_examples_from_repo_time_order(
    gh,
    repo_full_name,
    max_examples_per_repo,
    max_prs_to_scan,
    probe_prs=PROBE_PRS,
    probe_min=PROBE_MIN_EXAMPLES,
):
    try:
        repo = gh.get_repo(repo_full_name)
        print(f"[repo] {repo_full_name}", flush=True)
    except GithubException as e:
        print(f"[WARN] can't access {repo_full_name}: {e}", flush=True)
        return []

    collected = []
    pull_list = repo.get_pulls(state="closed", sort="updated", direction="desc")

    # ---- PHASE 1: probe first N PRs ----
    print(f"[probe] scanning first {probe_prs} PRs...", flush=True)
    for idx, pr in enumerate(pull_list):
        if idx >= probe_prs:
            break
        if len(collected) >= max_examples_per_repo:
            break
        print(f"  [PR #{pr.number}] scanning ...", flush=True)
        new_examples = scrape_examples_from_single_pr(repo, pr, max_needed=max_examples_per_repo - len(collected))
        if new_examples:
            collected.extend(new_examples)
            for ex in new_examples:
                print(f"     [OK] {ex['repo']} PR#{ex['pr_id']} {ex['file_path']}", flush=True)

    probe_count = len(collected)
    print(f"[probe done] {repo_full_name}: found {probe_count} examples in first {probe_prs} PRs", flush=True)

    if probe_count < probe_min:
        print(f"[skip] {repo_full_name}: low review signal (<{probe_min}), skipping deeper scan", flush=True)
        return collected[:max_examples_per_repo]

    # ---- PHASE 2: continue scanning until limits ----
    for idx, pr in enumerate(pull_list):
        if idx < probe_prs:
            continue
        if idx >= max_prs_to_scan or len(collected) >= max_examples_per_repo:
            break
        print(f"  [deep] PR #{pr.number} scanning ...", flush=True)
        new_examples = scrape_examples_from_single_pr(repo, pr, max_needed=max_examples_per_repo - len(collected))
        if new_examples:
            collected.extend(new_examples)
            for ex in new_examples:
                print(f"     [OK] {ex['repo']} PR#{ex['pr_id']} {ex['file_path']}", flush=True)

    print(f"[done] {repo_full_name}: total {len(collected)} examples", flush=True)
    return collected[:max_examples_per_repo]


########################
# main
########################

def main():
    token = load_token(TOKEN_PATH)
    gh = Github(token)
    repos_meta = load_repo_list(REPO_LIST_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        total = 0
        for repo_rec in repos_meta:
            full_name = repo_rec["full_name"]
            print(f"=== Collecting from {full_name} ===", flush=True)

            repo_examples = collect_examples_from_repo_time_order(
                gh,
                repo_full_name=full_name,
                max_examples_per_repo=MAX_EXAMPLES_PER_REPO,
                max_prs_to_scan=MAX_PRS_TO_SCAN,
            )

            for ex in repo_examples:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[write] wrote {len(repo_examples)} examples from {full_name}", flush=True)
            total += len(repo_examples)

        print(f"[TOTAL] {total} examples overall", flush=True)


if __name__ == "__main__":
    main()
