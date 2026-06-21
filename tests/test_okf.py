"""Tests for app/okf.py — OKF v0.1 validation and parsing."""
import pytest
from app.okf import (
    OKFDocument,
    OKFValidationError,
    parse_okf_markdown,
    validate_okf_bundle,
    extract_body_for_embedding,
    make_iso8601_timestamp,
)


class TestParseOKF:
    """Tests for parse_okf_markdown."""

    def test_parse_valid_minimal(self):
        """Minimal valid OKF document with only required type field."""
        content = """---
type: Decision
---
Body content here"""
        doc = parse_okf_markdown(content)
        assert doc.type == "Decision"
        assert doc.title is None
        assert doc.body == "Body content here"

    def test_parse_valid_full(self):
        """Full OKF document with all recommended fields."""
        content = """---
type: BigQuery Table
title: Customer Orders
description: One row per completed customer order.
resource: https://console.cloud.google.com/bigquery?p=acme&d=sales&t=orders
tags: [sales, orders, revenue]
timestamp: 2026-05-28T14:30:00Z
---

# Schema

| Column | Type | Description |
|--------|------|-------------|
| id | STRING | Unique ID |
"""
        doc = parse_okf_markdown(content)
        assert doc.type == "BigQuery Table"
        assert doc.title == "Customer Orders"
        assert doc.description == "One row per completed customer order."
        assert doc.resource == "https://console.cloud.google.com/bigquery?p=acme&d=sales&t=orders"
        assert doc.tags == ["sales", "orders", "revenue"]
        assert doc.timestamp == "2026-05-28T14:30:00Z"
        assert "# Schema" in doc.body

    def test_parse_with_extra_fields(self):
        """OKF document with producer-defined extra fields."""
        content = """---
type: Runbook
title: Incident Response
status: active
severity: P1
owner: platform-team
---
# Steps"""
        doc = parse_okf_markdown(content)
        assert doc.type == "Runbook"
        assert doc.extra["status"] == "active"
        assert doc.extra["severity"] == "P1"
        assert doc.extra["owner"] == "platform-team"

    def test_parse_missing_frontmatter(self):
        """Document without YAML frontmatter should raise error."""
        content = "Just a plain markdown file without frontmatter."
        with pytest.raises(OKFValidationError, match="Missing YAML frontmatter"):
            parse_okf_markdown(content)

    def test_parse_empty_type(self):
        """Document with empty type field should raise error."""
        content = """---
type: ""
---
Body"""
        with pytest.raises(OKFValidationError, match="Missing or empty required field"):
            parse_okf_markdown(content)

    def test_parse_missing_type(self):
        """Document without type field should raise error."""
        content = """---
title: Test
---
Body"""
        with pytest.raises(OKFValidationError, match="Missing or empty required field"):
            parse_okf_markdown(content)

    def test_parse_invalid_yaml(self):
        """Document with invalid YAML should raise error."""
        content = """---
type: Decision
invalid: [unclosed
---
Body"""
        with pytest.raises(OKFValidationError, match="Invalid YAML"):
            parse_okf_markdown(content)

    def test_parse_invalid_timestamp(self):
        """Document with invalid timestamp format should raise error."""
        content = """---
type: Decision
timestamp: not-a-date
---
Body"""
        with pytest.raises(OKFValidationError, match="Invalid ISO 8601 timestamp"):
            parse_okf_markdown(content)

    def test_parse_valid_timestamp_formats(self):
        """Various valid ISO 8601 timestamp formats."""
        timestamps = [
            ("2026-05-28T14:30:00Z", "2026-05-28T14:30:00Z"),
            ("2026-05-28T14:30:00+00:00", "2026-05-28T14:30:00Z"),  # Normalized to Z
            ("2026-05-28T14:30:00", "2026-05-28T14:30:00Z"),  # Z appended
            ("2026-05-28", "2026-05-28"),  # Date only, no timezone
        ]
        for input_ts, expected_ts in timestamps:
            content = f"""---
type: Decision
timestamp: {input_ts}
---
Body"""
            doc = parse_okf_markdown(content)
            assert doc.timestamp == expected_ts, f"Expected {expected_ts!r}, got {doc.timestamp!r} for input {input_ts!r}"

    def test_parse_tags_as_list(self):
        """Tags should be parsed as a list."""
        content = """---
type: Decision
tags: [python, fastapi, architecture]
---
Body"""
        doc = parse_okf_markdown(content)
        assert doc.tags == ["python", "fastapi", "architecture"]

    def test_parse_no_body(self):
        """Document with only frontmatter (no body) is valid."""
        content = """---
type: Decision
title: Empty
---"""
        doc = parse_okf_markdown(content)
        assert doc.type == "Decision"
        assert doc.body == ""


class TestOKFDocumentSerialization:
    """Tests for OKFDocument.to_markdown()."""

    def test_roundtrip(self):
        """Parsing and re-serializing should preserve content."""
        original = """---
type: Decision
title: Test Decision
description: A test
tags: [test]
timestamp: 2026-01-01T00:00:00Z
---

# Body

Some content here."""
        doc = parse_okf_markdown(original)
        serialized = doc.to_markdown()
        # Re-parse and verify
        doc2 = parse_okf_markdown(serialized)
        assert doc2.type == doc.type
        assert doc2.title == doc.title
        assert doc2.description == doc.description
        assert doc2.tags == doc.tags
        assert doc2.timestamp == doc.timestamp
        assert "Body" in doc2.body

    def test_frontmatter_property(self):
        """frontmatter property returns correct dict."""
        doc = OKFDocument(
            type="Decision",
            title="Test",
            description="A test",
            tags=["a", "b"],
            timestamp="2026-01-01T00:00:00Z",
            extra={"status": "active"},
        )
        fm = doc.frontmatter
        assert fm["type"] == "Decision"
        assert fm["title"] == "Test"
        assert fm["tags"] == ["a", "b"]
        assert fm["status"] == "active"


class TestExtractBody:
    """Tests for extract_body_for_embedding."""

    def test_extracts_body(self):
        """Should extract body after frontmatter."""
        content = """---
type: Decision
title: Test
---
# Body

Content here"""
        body = extract_body_for_embedding(content)
        assert "# Body" in body
        assert "Content here" in body
        assert "type:" not in body

    def test_no_frontmatter(self):
        """Should return full content if no frontmatter."""
        content = "Just plain text"
        assert extract_body_for_embedding(content) == "Just plain text"


class TestValidateBundle:
    """Tests for validate_okf_bundle."""

    def test_validate_empty_dir(self, tmp_path):
        """Empty directory should have no errors."""
        errors = validate_okf_bundle(str(tmp_path))
        assert len(errors) == 0

    def test_validate_valid_bundle(self, tmp_path):
        """Valid OKF bundle should have no errors."""
        # Create a valid concept document
        concept = tmp_path / "decisions"
        concept.mkdir()
        (concept / "test.md").write_text("""---
type: Decision
title: Test
---
Body""")
        # Create index.md (reserved, skipped)
        (concept / "index.md").write_text("# Index")
        errors = validate_okf_bundle(str(tmp_path))
        assert len(errors) == 0

    def test_validate_invalid_bundle(self, tmp_path):
        """Bundle with invalid concept should report errors."""
        concept = tmp_path / "decisions"
        concept.mkdir()
        (concept / "bad.md").write_text("No frontmatter here")
        errors = validate_okf_bundle(str(tmp_path))
        assert len(errors) == 1
        assert "Missing YAML frontmatter" in str(errors[0])


class TestMakeTimestamp:
    """Tests for make_iso8601_timestamp."""

    def test_valid_format(self):
        """Should return a valid ISO 8601 timestamp."""
        ts = make_iso8601_timestamp()
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts) >= 20
