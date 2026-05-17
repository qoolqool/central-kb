#!/usr/bin/env python3
"""kb CLI — Central KB command-line tool."""
import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Tuple

import httpx


CENTRAL_KB_URL_ENV = "CENTRAL_KB_URL"
CENTRAL_KB_PROJECT_ENV = "CENTRAL_KB_PROJECT"


def make_fqn(scope: str, namespace: str, key: str) -> str:
    return f"{scope}:{namespace}:{key}"


def parse_fqn(fqn: str) -> Tuple[str, str, str]:
    parts = fqn.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid FQN: {fqn}")
    return tuple(parts)


def build_central_url(env_url: Optional[str]) -> Optional[str]:
    if env_url:
        return env_url.rstrip("/")
    return None


def cmd_submit(args: argparse.Namespace):
    project = args.project or os.environ.get(CENTRAL_KB_PROJECT_ENV, "unknown")
    source = args.source or "local:cli"

    import sqlite3

    db_path = args.local_db
    if not db_path:
        # Auto-detect common local KB paths
        candidates = [
            "/project/.claude/agentdb.sqlite3",
            ".claude/agentdb.sqlite3",
            "agentdb.sqlite3",
        ]
        for c in candidates:
            if os.path.exists(c):
                db_path = c
                break
    if not db_path or not os.path.exists(db_path):
        print(f"Error: Local KB not found. Provide path with --local-db", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT key, namespace, content, metadata_json FROM embeddings"
    ).fetchall()
    conn.close()

    if not rows:
        print("No entries found in local KB.")
        return

    from app.dedup import simhash_64

    entries = []
    for key, namespace, content, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        title = meta.get("title", key)
        entries.append({
            "namespace": namespace,
            "key": key,
            "title": title,
            "content": content,
            "metadata": meta,
            "simhash": simhash_64(f"{title}\n{content}"[:1000]),
        })

    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    payload = {"project": project, "source": source, "entries": entries}
    resp = httpx.post(f"{url}/submit", json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    print(f"Submit to {project}:")
    print(f"  Accepted:   {data['accepted']}")
    print(f"  Duplicates: {data['duplicates']}")
    print(f"  Conflicted: {data['conflicted']}")
    for d in data.get("details", []):
        icons = {"accepted": "✓", "superseded_by": "→", "conflicted": "⚡", "error": "✗"}
        print(f"  {icons.get(d['status'], '?')} {d['fqn']} [{d['status']}]")
        if d.get("superseded_by"):
            print(f"     superseded by: {d['superseded_by']}")
        if d.get("conflict_id"):
            print(f"     conflict #{d['conflict_id']}")


def cmd_pull(args: argparse.Namespace):
    project = args.project or os.environ.get(CENTRAL_KB_PROJECT_ENV, "unknown")
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    scopes = ["own"]
    if args.global_scope:
        scopes.append("global")
    scope_str = ",".join(scopes)

    params = {
        "project": project,
        "after_version": args.after_version,
        "scope": scope_str,
    }

    resp = httpx.get(f"{url}/pull", params=params)
    resp.raise_for_status()
    data = resp.json()

    entries = data.get("entries", [])
    print(f"Pulled {len(entries)} entries from {project}:")
    for e in entries:
        tag = "[GLOBAL] " if e.get("scope") == "global" else ""
        print(f"  {tag}{e['fqn']} v{e['version']} — {e['title']}")

    drift = data.get("drift_warnings", [])
    if drift:
        print(f"\n⚠  DRIFT DETECTED ({len(drift)} items):")
        for d in drift:
            print(f"  Topic: {d.get('your_entry', '?')}")
            print(f"    You: {d.get('your_conclusion', '?')}")
            print(f"    vs:  {d.get('other_conclusion', '?')}")

    print(f"\nNext cursor: {data.get('next_cursor', '?')}")


def cmd_search(args: argparse.Namespace):
    query = " ".join(args.query)
    scope = args.scope or os.environ.get(CENTRAL_KB_PROJECT_ENV)
    namespace = args.namespace

    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if url:
        params = {"q": query, "limit": args.limit}
        if scope:
            params["scope"] = scope
        if namespace:
            params["namespace"] = namespace

        resp = httpx.get(f"{url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        print(f'Search: "{query}"')
        results = data.get("results", [])
        if not results:
            print("  No results.")
            return

        print(f"  Results: {len(results)} items:\n")
        for i, r in enumerate(results):
            print(f"  {i + 1}. [{r['namespace']}] {r['title']}  (score: {r['score']:.4f})")
            print(f"     {r['fqn']}")
            print(f"     {r['content'][:120]}...")
            print()
    else:
        script = os.path.join(os.path.dirname(__file__), "..", "..",
                              "tooling", "scripts", "search-kb-memory.py")
        cmd = [sys.executable, script] + list(args.query)
        if namespace:
            cmd.extend(["-n", namespace])
        if args.limit:
            cmd.extend(["-l", str(args.limit)])
        subprocess.run(cmd)


def cmd_drift(args: argparse.Namespace):
    project = args.project or os.environ.get(CENTRAL_KB_PROJECT_ENV, "unknown")
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    resp = httpx.get(f"{url}/drift", params={"project": project})
    resp.raise_for_status()
    data = resp.json()

    items = data.get("drift_items", [])
    if not items:
        print(f"No drift detected for {project}.")
        return

    print(f"Drift report for {project}:")
    for d in items:
        print(f"\n  ⚠  Topic similarity: {d.get('topic_similarity', 0):.2f}")
        your = d.get("your_entry", {})
        other = d.get("conflicting_entry", {})
        print(f"     You: {your.get('fqn', '?')} — {your.get('title', '?')}")
        print(f"     vs:  {other.get('fqn', '?')} — {other.get('title', '?')}")


def cmd_candidates(args: argparse.Namespace):
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    resp = httpx.get(f"{url}/candidates")
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        print("No promotion candidates.")
        return

    print(f"Promotion candidates ({len(candidates)}):")
    for c in candidates:
        print(f"  #{c['id']} — {c['candidate_fqn']}")
        print(f"     Matches: {c['project_count']} projects, avg similarity: {c['avg_similarity']:.2f}")
        print(f"     Status: {c['status']}")


def cmd_promote(args: argparse.Namespace):
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    payload = {
        "candidate_id": args.candidate_id,
        "action": args.action,
        "verdict_by": os.environ.get("USER", "unknown"),
    }
    resp = httpx.post(f"{url}/promote", json=payload)
    resp.raise_for_status()
    print(f"Candidate #{args.candidate_id}: {args.action}d.")


def cmd_conflicts(args: argparse.Namespace):
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    resp = httpx.get(f"{url}/conflicts")
    resp.raise_for_status()
    data = resp.json()

    conflicts = data.get("conflicts", [])
    if not conflicts:
        print("No pending conflicts.")
        return

    print(f"Pending conflicts ({len(conflicts)}):")
    for c in conflicts:
        print(f"  #{c['id']}: {c['existing_fqn']} ← {c['proposed_fqn']}")
        print(f"     Proposed: {c.get('proposed_content', '')[:80]}...")


def cmd_resolve(args: argparse.Namespace):
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    payload = {"resolution": args.resolution}
    resp = httpx.post(f"{url}/conflicts/{args.conflict_id}/resolve", json=payload)
    resp.raise_for_status()
    print(f"Conflict #{args.conflict_id}: resolved ({args.resolution}).")


def main():
    parser = argparse.ArgumentParser(
        description="Central KB — cross-project knowledge base CLI"
    )
    parser.add_argument("--central-url", help="Override CENTRAL_KB_URL")
    sub = parser.add_subparsers(dest="command")

    p_submit = sub.add_parser("submit", help="Submit local KB to central server")
    p_submit.add_argument("--project", "-p", help="Project namespace")
    p_submit.add_argument("--source", "-s", default="local:cli")
    p_submit.add_argument("--local-db",
                          help="Path to local KB SQLite database (default: auto-detect)")
    p_submit.set_defaults(func=cmd_submit)

    p_pull = sub.add_parser("pull", help="Pull entries from central KB")
    p_pull.add_argument("--project", "-p", help="Project namespace")
    p_pull.add_argument("--after-version", type=int, default=0)
    p_pull.add_argument("--global", dest="global_scope", action="store_true",
                        help="Include global namespace")
    p_pull.set_defaults(func=cmd_pull)

    p_search = sub.add_parser("search", help="Search across namespaces")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--scope", "-s", help="Scope filter (project name)")
    p_search.add_argument("--namespace", "-n", help="Namespace filter")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_drift = sub.add_parser("drift", help="Show drift report")
    p_drift.add_argument("--project", "-p", help="Project namespace")
    p_drift.set_defaults(func=cmd_drift)

    p_candidates = sub.add_parser("candidates", help="List promotion candidates")
    p_candidates.set_defaults(func=cmd_candidates)

    p_promote = sub.add_parser("promote", help="Approve/reject promotion candidate")
    p_promote.add_argument("candidate_id", type=int)
    p_promote.add_argument("action", choices=["approve", "reject"])
    p_promote.set_defaults(func=cmd_promote)

    p_conflicts = sub.add_parser("conflicts", help="List pending conflicts")
    p_conflicts.set_defaults(func=cmd_conflicts)

    p_resolve = sub.add_parser("resolve", help="Resolve a conflict")
    p_resolve.add_argument("conflict_id", type=int)
    p_resolve.add_argument("resolution", choices=["keep_existing", "accept_proposed", "merge_manual"])
    p_resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args()

    if args.central_url:
        os.environ[CENTRAL_KB_URL_ENV] = args.central_url

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
