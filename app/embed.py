#!/usr/bin/env python3
"""Embedding client — connects to the embedding server HTTP API.

The embed server (scripts/embed-server.py) runs as a sidecar container
and provides embeddings via both HTTP API (port 9001) and Unix socket.
All containers share the same embed server, avoiding duplicate model loading.

Fallback chain: HTTP embed server → Unix socket → Ollama (dev only).
"""
import json
import os
import socket
import struct
import urllib.request
from typing import Optional

# Configurable via environment
EMBED_SERVER_URL = os.environ.get(
    "EMBED_SERVER_URL",
    "http://embed-server:9001",  # Docker DNS name for the sidecar
)
EMBED_SOCK = os.environ.get("EMBED_SOCK_PATH", "/tmp/embed-server.sock")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embeddings")
OLLAMA_MODEL = "bge-large:latest"
VEC_DIM = 1024


def pack_vector(vec: list[float]) -> bytes:
    """Pack float list into compact binary for SQLite BLOB."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    """Unpack binary BLOB back into float list."""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def embed_via_http(text: str) -> Optional[list[float]]:
    """Embed via the shared HTTP embed server (~150ms)."""
    try:
        data = json.dumps({"text": text[:512]}).encode("utf-8")
        url = f"{EMBED_SERVER_URL}/embed"
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        if "error" in result:
            return None
        return result["embedding"]
    except Exception:
        return None


def embed_via_socket(text: str) -> Optional[list[float]]:
    """Try the Unix socket daemon (~40ms)."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(EMBED_SOCK)
        sock.sendall(text[:512].encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        sock.close()
        result = json.loads(b"".join(chunks))
        if "error" in result:
            return None
        return result["embedding"]
    except Exception:
        return None


def embed_via_ollama(text: str) -> list[float]:
    """Fallback via Ollama HTTP — ~330ms."""
    data = json.dumps({"model": OLLAMA_MODEL, "prompt": text[:256]}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["embedding"]


def embed_text(text: str) -> Optional[list[float]]:
    """Embed text. Tries HTTP embed server → Unix socket → Ollama fallback.

    Returns None if all methods fail.
    """
    result = embed_via_http(text)
    if result is not None:
        return result
    result = embed_via_socket(text)
    if result is not None:
        return result
    try:
        return embed_via_ollama(text)
    except Exception:
        return None
