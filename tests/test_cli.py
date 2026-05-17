"""Tests for kb_cli/cli.py."""
import pytest


def test_make_fqn():
    from kb_cli.cli import make_fqn
    assert make_fqn("x402-poc", "decisions", "DEC-001") == "x402-poc:decisions:DEC-001"


def test_parse_fqn():
    from kb_cli.cli import parse_fqn
    assert parse_fqn("x402-poc:decisions:DEC-001") == ("x402-poc", "decisions", "DEC-001")


def test_build_url():
    from kb_cli.cli import build_central_url
    assert build_central_url(None) is None
    assert build_central_url("http://localhost:9000") == "http://localhost:9000"
    assert build_central_url("http://localhost:9000/") == "http://localhost:9000"
