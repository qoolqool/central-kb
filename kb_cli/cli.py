#!/usr/bin/env python3
"""kb CLI — Central KB command-line tool.

Supports both legacy and OKF (Open Knowledge Format) submission.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import httpx
import yaml


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


def _make_key_from_title(title: str) -> str:
    """Convert a title to a URL-safe key."""
    key = title.lower().strip()
    key = re.sub(r"[^a-z0-9]+", "-", key)
    key = key.strip("-")
    return key[:100] or "untitled"


def _map_type_to_namespace(okf_type: str) -> str:
    """Map an OKF type to a namespace for storage."""
    type_lower = okf_type.lower()
    if "decision" in type_lower:
        return "decisions"
    elif "pattern" in type_lower or "playbook" in type_lower or "runbook" in type_lower:
        return "patterns"
    elif "session" in type_lower:
        return "sessions"
    elif "metric" in type_lower:
        return "metrics"
    elif "table" in type_lower or "dataset" in type_lower:
        return "tables"
    else:
        return "concepts"


def _validate_okf_markdown(content: str) -> dict:
    """Validate OKF markdown and return parsed frontmatter.

    Returns dict with 'valid' bool, 'errors' list, and 'frontmatter' dict.
    """
    result = {"valid": True, "errors": [], "frontmatter": {}, "body": ""}

    # Check for YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not fm_match:
        result["valid"] = False
        result["errors"].append("Missing YAML frontmatter block (must start with '---' and end with '---')")
        return result

    fm_raw = fm_match.group(1)
    body = fm_match.group(2).strip()

    try:
        fm = yaml.safe_load(fm_raw)
    except yaml.YAMLError as e:
        result["valid"] = False
        result["errors"].append(f"Invalid YAML frontmatter: {e}")
        return result

    if not isinstance(fm, dict):
        result["valid"] = False
        result["errors"].append("Frontmatter must be a YAML mapping")
        return result

    # Check required 'type' field
    type_val = fm.get("type")
    if not type_val or not isinstance(type_val, str) or not type_val.strip():
        result["valid"] = False
        result["errors"].append("Missing or empty required field: 'type'")

    result["frontmatter"] = fm
    result["body"] = body
    return result


def cmd_submit(args: argparse.Namespace):
    project = args.project or os.environ.get(CENTRAL_KB_PROJECT_ENV, "unknown")
    source = args.source or "local:cli"

    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    # Determine input source
    okf_entries = []

    if args.okf_dir:
        # Read OKF markdown files from directory
        kb_dir = Path(args.okf_dir)
        if not kb_dir.is_dir():
            print(f"Error: OKF directory not found: {kb_dir}", file=sys.stderr)
            sys.exit(1)

        for md_file in sorted(kb_dir.rglob("*.md")):
            # Skip reserved filenames
            if md_file.name in ("index.md", "log.md"):
                continue
            try:
                content = md_file.read_text()
                validation = _validate_okf_markdown(content)
                if not validation["valid"]:
                    print(f"  ⚠  Skipping {md_file.relative_to(kb_dir)}: "
                          f"{'; '.join(validation['errors'])}", file=sys.stderr)
                    continue
                okf_entries.append({
                    "markdown": content,
                })
                print(f"  ✓ {md_file.relative_to(kb_dir)}")
            except Exception as e:
                print(f"  ✗ {md_file.relative_to(kb_dir)}: {e}", file=sys.stderr)

        if not okf_entries:
            print("No valid OKF entries found.")
            return

    elif args.local_db:
        # Legacy: read from local SQLite
        import sqlite3
        db_path = args.local_db
        if not os.path.exists(db_path):
            print(f"Error: Local KB not found: {db_path}", file=sys.stderr)
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
        return

    else:
        # Auto-detect: try OKF directory first, then legacy SQLite
        okf_candidates = [
            Path("/project/knowledgebase"),
            Path("knowledgebase"),
        ]
        for c in okf_candidates:
            if c.is_dir() and list(c.rglob("*.md")):
                args.okf_dir = str(c)
                print(f"Auto-detected OKF directory: {c}")
                cmd_submit(args)
                return

        # Fall back to legacy auto-detect
        db_candidates = [
            "/project/.agent/agentdb.sqlite3",
            ".claude/agentdb.sqlite3",
            "agentdb.sqlite3",
        ]
        for c in db_candidates:
            if os.path.exists(c):
                args.local_db = c
                print(f"Auto-detected local DB: {c}")
                cmd_submit(args)
                return

        print("Error: No knowledge source found. Use --okf-dir or --local-db.", file=sys.stderr)
        sys.exit(1)

    # Submit OKF entries
    payload = {
        "project": project,
        "source": source,
        "okf_entries": okf_entries,
    }
    resp = httpx.post(f"{url}/submit", json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()

    print(f"\nSubmit to {project}:")
    print(f"  Accepted:   {data['accepted']}")
    print(f"  Duplicates: {data['duplicates']}")
    print(f"  Conflicted: {data['conflicted']}")
    for d in data.get("details", []):
        icons = {"accepted": "✓", "auto_merged": "→", "superseded_by": "→",
                 "conflicted": "⚡", "error": "✗"}
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
            okf_type = r.get("okf_type") or r.get("namespace", "?")
            okf_tags = r.get("okf_tags") or []
            tag_str = f" [{', '.join(okf_tags[:3])}]" if okf_tags else ""
            print(f"  {i + 1}. [{okf_type}]{tag_str} {r['title']}  (score: {r['score']:.4f})")
            print(f"     {r['fqn']}")
            if r.get("okf_description"):
                print(f"     {r['okf_description'][:120]}")
            else:
                print(f"     {r['content'][:120]}...")
            print()
    else:
        script = os.path.join(os.path.dirname(__file__), "..", "..",
                              "scripts", "search-kb-memory.py")
        cmd = [sys.executable, script] + list(args.query)
        if namespace:
            cmd.extend(["-n", namespace])
        if args.limit:
            cmd.extend(["-l", str(args.limit)])
        subprocess.run(cmd)


def cmd_convert(args: argparse.Namespace):
    """Convert existing knowledgebase entries to OKF format."""
    from app.okf import make_iso8601_timestamp

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    errors = 0

    # Walk through input directory
    for yaml_file in sorted(input_dir.rglob("*.yaml")):
        rel_path = yaml_file.relative_to(input_dir)
        try:
            content = yaml_file.read_text()
            parsed = _parse_yaml_kb_entry(content)
            if parsed is None:
                print(f"  ⚠  Skipping {rel_path}: could not parse")
                errors += 1
                continue

            # Determine output path
            namespace = parsed.get("namespace", "concepts")
            key = parsed.get("key") or _make_key_from_title(parsed.get("title", "untitled"))
            out_subdir = output_dir / namespace
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_path = out_subdir / f"{key}.md"

            # Build OKF markdown
            okf_content = _build_okf_markdown(parsed)
            out_path.write_text(okf_content)

            print(f"  ✓ {rel_path} → {out_path.relative_to(output_dir)}")
            converted += 1

        except Exception as e:
            print(f"  ✗ {rel_path}: {e}", file=sys.stderr)
            errors += 1

    # Generate index.md files
    _generate_index_files(output_dir)

    print(f"\nConverted: {converted}, Errors: {errors}")
    print(f"Output: {output_dir}")


def _parse_yaml_kb_entry(content: str) -> Optional[dict]:
    """Parse a legacy YAML knowledgebase entry."""
    # Try to extract YAML frontmatter (between --- markers)
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if fm_match:
        fm_raw = fm_match.group(1)
        body = fm_match.group(2).strip()
    else:
        # No frontmatter — treat entire content as body
        fm_raw = content
        body = ""

    try:
        fm = yaml.safe_load(fm_raw)
    except yaml.YAMLError:
        return None

    if not isinstance(fm, dict):
        return None

    return {
        "frontmatter": fm,
        "body": body,
    }


def _build_okf_markdown(parsed: dict) -> str:
    """Build OKF markdown from parsed legacy entry."""
    fm = parsed.get("frontmatter", {})
    body = parsed.get("body", "")

    # Map fields
    okf_type = _map_legacy_type(fm)
    title = fm.get("title", fm.get("name", fm.get("id", "Untitled")))
    description = fm.get("description", fm.get("summary", fm.get("decision", "")))
    resource = fm.get("resource", fm.get("source", ""))
    tags = fm.get("topics", fm.get("tags", []))
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    timestamp = fm.get("timestamp", fm.get("date", fm.get("created_at", "")))
    if timestamp and not _is_iso8601(timestamp):
        # Try to convert common date formats
        timestamp = _convert_to_iso8601(timestamp)

    # Build body from remaining fields
    body_parts = []
    if body:
        body_parts.append(body)

    # Add conventional sections for known fields
    for section_name in ["context", "consequences", "implementation", "rationale", "alternatives"]:
        val = fm.get(section_name)
        if val:
            section_title = section_name.capitalize()
            body_parts.append(f"\n# {section_title}\n\n{val}")

    body_text = "\n".join(body_parts).strip()

    # Build frontmatter
    lines = ["---"]
    lines.append(f"type: {okf_type}")
    lines.append(f"title: {title}")
    if description:
        lines.append(f"description: {description}")
    if resource:
        lines.append(f"resource: {resource}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    if timestamp:
        lines.append(f"timestamp: {timestamp}")

    # Preserve extra fields
    standard_keys = {"type", "title", "description", "resource", "tags", "timestamp",
                     "topics", "date", "name", "id", "summary", "decision", "source",
                     "context", "consequences", "implementation", "rationale", "alternatives",
                     "created_at", "status", "category"}
    for k, v in fm.items():
        if k not in standard_keys and v is not None:
            if isinstance(v, str) and "\n" in v:
                lines.append(f"{k}: |")
                for line in v.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{k}: {v}")

    lines.append("---")

    if body_text:
        lines.append("")
        lines.append(body_text)

    return "\n".join(lines)


def _map_legacy_type(fm: dict) -> str:
    """Map legacy category/type to OKF type."""
    category = fm.get("category", "").lower()
    type_val = fm.get("type", "").lower()

    if type_val:
        return type_val.capitalize()
    elif "decision" in category or "arch" in category:
        return "Decision"
    elif "pattern" in category:
        return "Pattern"
    elif "session" in category:
        return "Session"
    elif "metric" in category:
        return "Metric"
    elif "runbook" in category or "playbook" in category:
        return "Runbook"
    elif "table" in category or "dataset" in category:
        return "Table"
    else:
        return "Concept"


def _is_iso8601(ts: str) -> bool:
    """Check if a timestamp is already ISO 8601."""
    patterns = [
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$",
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
        r"^\d{4}-\d{2}-\d{2}$",
    ]
    return any(re.match(p, ts) for p in patterns)


def _convert_to_iso8601(date_str: str) -> str:
    """Convert common date formats to ISO 8601."""
    # Try YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00Z"
    # Try MM/DD/YYYY
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})", date_str)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}T00:00:00Z"
    return date_str


def _generate_index_files(output_dir: Path):
    """Generate index.md files for each subdirectory."""
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        md_files = sorted(subdir.glob("*.md"))
        concept_files = [f for f in md_files if f.name not in ("index.md", "log.md")]
        if not concept_files:
            continue

        index_path = subdir / "index.md"
        lines = [f"# {subdir.name.capitalize()}", ""]
        for cf in concept_files:
            try:
                content = cf.read_text()
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                if fm_match:
                    fm = yaml.safe_load(fm_match.group(1)) or {}
                    title = fm.get("title", cf.stem)
                    desc = fm.get("description", "")
                    lines.append(f"* [{title}]({cf.name}) - {desc}" if desc else f"* [{title}]({cf.name})")
                else:
                    lines.append(f"* [{cf.stem}]({cf.name})")
            except Exception:
                lines.append(f"* [{cf.stem}]({cf.name})")

        index_path.write_text("\n".join(lines) + "\n")
        print(f"  📋 Generated {index_path.relative_to(output_dir)}")


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


def cmd_validate(args: argparse.Namespace):
    """Validate OKF compliance of a bundle directory."""
    from app.okf import validate_okf_bundle

    bundle_path = Path(args.bundle_dir)
    if not bundle_path.is_dir():
        print(f"Error: Bundle directory not found: {bundle_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating OKF bundle: {bundle_path}")
    errors = validate_okf_bundle(str(bundle_path))

    if not errors:
        print("✅ Bundle is OKF v0.1 conformant!")
        return

    print(f"\nFound {len(errors)} validation error(s):")
    for e in errors:
        field = f" ({e.field})" if e.field else ""
        print(f"  ✗ {e}{field}")


def cmd_health(args: argparse.Namespace):
    """Check Central KB server health."""
    url = build_central_url(os.environ.get(CENTRAL_KB_URL_ENV))
    if not url:
        print("Error: CENTRAL_KB_URL not set.", file=sys.stderr)
        sys.exit(1)

    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        print(f"Central KB: {data.get('status', 'unknown')}")
        print(f"  Version: {data.get('version', '?')}")
    except Exception as e:
        print(f"Error: Cannot reach Central KB at {url}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Central KB — cross-project knowledge base CLI (OKF v0.1)"
    )
    parser.add_argument("--central-url", help="Override CENTRAL_KB_URL")
    sub = parser.add_subparsers(dest="command")

    p_submit = sub.add_parser("submit", help="Submit local KB to central server")
    p_submit.add_argument("--project", "-p", help="Project namespace")
    p_submit.add_argument("--source", "-s", default="local:cli")
    p_submit.add_argument("--local-db",
                          help="Path to local KB SQLite database (legacy)")
    p_submit.add_argument("--okf-dir",
                          help="Path to OKF markdown directory (default: auto-detect)")
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

    p_convert = sub.add_parser("convert", help="Convert legacy KB entries to OKF format")
    p_convert.add_argument("input_dir", help="Input directory with legacy YAML files")
    p_convert.add_argument("output_dir", help="Output directory for OKF markdown files")
    p_convert.set_defaults(func=cmd_convert)

    p_validate = sub.add_parser("validate", help="Validate OKF bundle compliance")
    p_validate.add_argument("bundle_dir", help="OKF bundle directory to validate")
    p_validate.set_defaults(func=cmd_validate)

    p_health = sub.add_parser("health", help="Check Central KB server health")
    p_health.set_defaults(func=cmd_health)

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
