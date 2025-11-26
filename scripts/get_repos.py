from github import Github
import json

# read your token
with open("github_token.txt", "r", encoding="utf-8") as f:
    token = f.read().strip()

g = Github(token)

results = []

# 1. search for top Python repos by stars
#    we'll overfetch (like top 100) then filter
query = 'language:Python sort:stars'
repos = g.search_repositories(query=query)  # already sorted by stars desc

for repo in repos[500:1500]:
    print(f"Processing repo: {repo.full_name} with {repo.stargazers_count} stars")
    try:
        # skip forks
        if repo.fork:
            continue

        # primary language check
        if (repo.language or "").lower() != "python":
            continue

        # count contributors
        contributors = repo.get_contributors()
        contributor_count = contributors.totalCount

        # count PRs (open + closed)
        pulls_open = repo.get_pulls(state="open").totalCount
        pulls_closed = repo.get_pulls(state="closed").totalCount
        total_prs = pulls_open + pulls_closed

        # apply thresholds
        if total_prs >= 1000 and contributor_count >= 50:
            results.append({
                "full_name": repo.full_name,
                "stars": repo.stargazers_count,
                "total_prs": total_prs,
                "contributors": contributor_count,
            })

    except Exception as e:
        # some repos may have perms/rate limit weirdness, just skip
        continue

# sort the passing repos again by stars desc
results_sorted = sorted(results, key=lambda r: r["stars"], reverse=True)

# Write results to a file
with open("dataset/top_python_repos.jsonl", "w", encoding="utf-8") as f:
    for r in results_sorted:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
