#!/usr/bin/env python3
"""Seed tool — import an existing local KB into the central server."""
import argparse
import json
import os
import sqlite3
import sys

import httpx

from app.dedup import simhash_64


def collect_local_entries(db_path: str) -> list[dict]:
    """Read entries from a local KB SQLite database."""
    if not os.path.exists(db_path):
        print(f"Error: Local KB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row[0] for row in tables}

    if "embeddings" in table_names:
        rows = conn.execute(
            "SELECT key, namespace, content, metadata_json FROM embeddings"
        ).fetchall()
    elif "entries" in table_names:
        rows = conn.execute(
            "SELECT key, namespace, content, metadata_json FROM entries"
        ).fetchall()
    else:
        print("Error: Unknown KB schema — expected 'embeddings' or 'entries' table",
              file=sys.stderr)
        sys.exit(1)

    conn.close()

    entries = []
    for key, namespace, content, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        title = meta.get("title", key)

        entries.append({
            "namespace": namespace or "decisions",
            "key": key,
            "title": title,
            "content": content,
            "metadata": meta,
            "simhash": simhash_64(f"{title}\n{content}"[:1000]),
        })

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Import a local KB into the central server"
    )
    parser.add_argument("--project", "-p", required=True, help="Project namespace")
    parser.add_argument("--from", dest="from_db", required=True,
                        help="Path to local KB SQLite database")
    parser.add_argument("--source", default="local:seed", help="Source identifier")
    parser.add_argument("--central-url",
                        default=os.environ.get("CENTRAL_KB_URL", "http://localhost:9000"),
                        help="Central KB server URL")
    args = parser.parse_args()

    entries = collect_local_entries(args.from_db)
    if not entries:
        print("No entries found in local KB.")
        return

    print(f"Collected {len(entries)} entries from {args.from_db}")
    print(f"Submitting to {args.central_url} as project '{args.project}'...")

    batch_size = 50
    total_accepted = 0
    total_duplicates = 0
    total_conflicted = 0

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        payload = {"project": args.project, "source": args.source, "entries": batch}

        resp = httpx.post(f"{args.central_url}/submit", json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

        total_accepted += data["accepted"]
        total_duplicates += data["duplicates"]
        total_conflicted += data["conflicted"]

        if data.get("conflict_ids"):
            print(f"  Conflicts: #{', #'.join(str(c) for c in data['conflict_ids'])}")
        print(f"  Batch {i // batch_size + 1}: {len(batch)} entries → "
              f"{data['accepted']} accepted, {data['duplicates']} dupes, "
              f"{data['conflicted']} conflicts")

    print(f"\nDone. Total: {total_accepted} accepted, "
          f"{total_duplicates} duplicates, {total_conflicted} conflicts")


if __name__ == "__main__":
    main()
