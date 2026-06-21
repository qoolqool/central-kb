"""Open Knowledge Format (OKF) v0.1 — validation and parsing utilities.

OKF spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

Conformance (v0.1):
1. Every non-reserved .md file contains a parseable YAML frontmatter block.
2. Every frontmatter block contains a non-empty `type` field.
3. Reserved filenames (index.md, log.md) follow spec structure when present.
"""
import re
import yaml
from datetime import datetime, timezone
from typing import Any, Optional

# Reserved filenames that MUST NOT be used for concept documents
RESERVED_FILENAMES = {"index.md", "log.md"}

# Recommended frontmatter fields (in priority order per spec)
RECOMMENDED_FIELDS = ["title", "description", "resource", "tags", "timestamp"]

# Conventional body section headings
CONVENTIONAL_SECTIONS = ["Schema", "Examples", "Citations"]


class OKFValidationError(Exception):
    """Raised when an OKF document fails validation."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class OKFDocument:
    """Represents a parsed OKF concept document."""

    def __init__(
        self,
        type: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        resource: Optional[str] = None,
        tags: Optional[list[str]] = None,
        timestamp: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        body: str = "",
        source_path: Optional[str] = None,
    ):
        self.type = type
        self.title = title
        self.description = description
        self.resource = resource
        self.tags = tags or []
        self.timestamp = timestamp
        self.extra = extra or {}
        self.body = body
        self.source_path = source_path

    @property
    def frontmatter(self) -> dict[str, Any]:
        """Return the full frontmatter dict (required + recommended + extra)."""
        fm: dict[str, Any] = {"type": self.type}
        if self.title is not None:
            fm["title"] = self.title
        if self.description is not None:
            fm["description"] = self.description
        if self.resource is not None:
            fm["resource"] = self.resource
        if self.tags:
            fm["tags"] = self.tags
        if self.timestamp is not None:
            fm["timestamp"] = self.timestamp
        fm.update(self.extra)
        return fm

    def to_markdown(self) -> str:
        """Serialize the document back to OKF markdown."""
        lines = ["---"]
        lines.append(f"type: {self.type}")
        if self.title:
            lines.append(f"title: {self.title}")
        if self.description:
            # Use block scalar for multi-line descriptions
            if "\n" in self.description:
                lines.append(f"description: |")
                for line in self.description.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"description: {self.description}")
        if self.resource:
            lines.append(f"resource: {self.resource}")
        if self.tags:
            lines.append(f"tags: [{', '.join(self.tags)}]")
        if self.timestamp:
            lines.append(f"timestamp: {self.timestamp}")
        for k, v in self.extra.items():
            if isinstance(v, str) and "\n" in v:
                lines.append(f"{k}: |")
                for line in v.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("---")
        if self.body:
            lines.append("")
            lines.append(self.body)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"OKFDocument(type={self.type!r}, title={self.title!r})"


def parse_okf_markdown(content: str, source_path: Optional[str] = None) -> OKFDocument:
    """Parse an OKF markdown string into an OKFDocument.

    Args:
        content: Full markdown content (frontmatter + body).
        source_path: Optional file path for error messages.

    Returns:
        Parsed OKFDocument.

    Raises:
        OKFValidationError: If the document is not valid OKF.
    """
    # Split frontmatter from body
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not fm_match:
        raise OKFValidationError(
            "Missing YAML frontmatter block (must start with '---' and end with '---')"
        )

    fm_raw = fm_match.group(1)
    body = fm_match.group(2).strip()

    # Parse YAML frontmatter
    try:
        fm = yaml.safe_load(fm_raw)
    except yaml.YAMLError as e:
        raise OKFValidationError(f"Invalid YAML frontmatter: {e}")

    if not isinstance(fm, dict):
        raise OKFValidationError("Frontmatter must be a YAML mapping (key-value pairs)")

    # Validate required 'type' field
    type_val = fm.get("type")
    if not type_val or not isinstance(type_val, str) or not type_val.strip():
        raise OKFValidationError(
            "Missing or empty required field: 'type'",
            field="type"
        )

    # Extract recommended fields
    title = fm.get("title")
    if title is not None and not isinstance(title, str):
        raise OKFValidationError("'title' must be a string", field="title")

    description = fm.get("description")
    if description is not None and not isinstance(description, str):
        raise OKFValidationError("'description' must be a string", field="description")

    resource = fm.get("resource")
    if resource is not None and not isinstance(resource, str):
        raise OKFValidationError("'resource' must be a string", field="resource")

    tags = fm.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            raise OKFValidationError("'tags' must be a list", field="tags")
        tags = [str(t) for t in tags]

    timestamp = fm.get("timestamp")
    if timestamp is not None:
        # YAML may parse ISO 8601 as datetime object
        if hasattr(timestamp, "isoformat"):
            ts_str = timestamp.isoformat()
            # Check if it's a date-only (no time component)
            if "T" not in ts_str:
                timestamp = ts_str  # Keep as date-only
            else:
                # Normalize +00:00 to Z for consistency
                if ts_str.endswith("+00:00"):
                    timestamp = ts_str[:-6] + "Z"
                elif "+" not in ts_str and not ts_str.endswith("Z"):
                    timestamp = ts_str + "Z"
                else:
                    timestamp = ts_str
        if not isinstance(timestamp, str):
            raise OKFValidationError("'timestamp' must be a string", field="timestamp")
        # Validate ISO 8601 format
        _validate_iso8601(timestamp)

    # Collect extra fields (everything not in the standard set)
    standard_keys = {"type", "title", "description", "resource", "tags", "timestamp"}
    extra = {k: v for k, v in fm.items() if k not in standard_keys}

    return OKFDocument(
        type=type_val.strip(),
        title=title.strip() if isinstance(title, str) else None,
        description=description.strip() if isinstance(description, str) else None,
        resource=resource.strip() if isinstance(resource, str) else None,
        tags=tags,
        timestamp=timestamp.strip() if isinstance(timestamp, str) else None,
        extra=extra,
        body=body,
        source_path=source_path,
    )


def _validate_iso8601(ts: str) -> None:
    """Validate ISO 8601 datetime format (accepts with/without timezone)."""
    # Accept formats like:
    # 2026-05-28T14:30:00Z
    # 2026-05-28T14:30:00+00:00
    # 2026-05-28T14:30:00
    # 2026-05-28 (date only)
    patterns = [
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$",
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
        r"^\d{4}-\d{2}-\d{2}$",
    ]
    if not any(re.match(p, ts) for p in patterns):
        raise OKFValidationError(
            f"Invalid ISO 8601 timestamp: {ts!r}. "
            f"Expected format: YYYY-MM-DDTHH:MM:SSZ",
            field="timestamp"
        )


def validate_okf_bundle(bundle_path: str) -> list[OKFValidationError]:
    """Validate an entire OKF bundle directory.

    Checks:
    1. Every non-reserved .md file has parseable YAML frontmatter with non-empty type.
    2. Reserved filenames (index.md, log.md) are not used for concept documents.

    Returns:
        List of validation errors (empty if bundle is conformant).
    """
    import os
    errors = []

    for root, dirs, files in os.walk(bundle_path):
        for fname in files:
            if not fname.endswith(".md"):
                continue

            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, bundle_path)

            if fname in RESERVED_FILENAMES:
                # Reserved files are not concept documents — skip validation
                continue

            try:
                with open(fpath, "r") as f:
                    content = f.read()
                parse_okf_markdown(content, source_path=rel_path)
            except OKFValidationError as e:
                errors.append(e)
            except IOError as e:
                errors.append(OKFValidationError(f"Cannot read {rel_path}: {e}"))

    return errors


def extract_body_for_embedding(content: str) -> str:
    """Extract the body portion of an OKF document for embedding.

    Strips the YAML frontmatter and returns only the markdown body.
    This ensures embeddings are generated from content, not metadata.
    """
    fm_match = re.match(r"^---\s*\n.*?\n---\s*\n?(.*)", content, re.DOTALL)
    if fm_match:
        return fm_match.group(1).strip()
    return content.strip()


def make_iso8601_timestamp() -> str:
    """Generate a current ISO 8601 timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
