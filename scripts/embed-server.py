#!/usr/bin/env python3
"""
Embedding HTTP API server — loads bge-large-en-v1.5 via sentence-transformers
once, serves embeddings via HTTP. All containers (central-kb, toy-rag, etc.)
can share this single server instead of each pulling Ollama/sentence-transformers.

Usage:
    python3 scripts/embed-server.py [--host 0.0.0.0] [--port 9001]

Endpoints:
    POST /embed    — {"text": "..."} → {"embedding": [...], "dim": 1024}
    POST /batch    — {"texts": ["...", "..."]} → {"embeddings": [[...], ...]}
    GET  /health   — {"status": "ok", "model": "BAAI/bge-large-en-v1.5", ...}
"""
import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
VEC_DIM = 1024
model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        raise RuntimeError("Model is still loading — try again in a few seconds")
    return model


def _background_load_model():
    """Load the model in the background so HTTP server starts immediately."""
    global model
    import time as _time
    print(f"[embed-server] Loading model {MODEL_NAME} in background...", file=sys.stderr, flush=True)
    t0 = _time.time()
    try:
        m = SentenceTransformer(MODEL_NAME)
        model = m
        print(f"[embed-server] Model loaded in {_time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[embed-server] ERROR loading model: {e}", file=sys.stderr, flush=True)


class EmbedHandler(BaseHTTPRequestHandler):
    """HTTP handler for embedding requests."""

    def log_message(self, fmt, *args):
        try:
            if args and len(args) >= 2 and int(args[1]) >= 400:
                super().log_message(fmt, *args)
        except (ValueError, IndexError):
            pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Optional[dict]:
        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            return None
        return json.loads(self.rfile.read(cl).decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "model": MODEL_NAME,
                "dim": VEC_DIM,
                "server": "central-kb-embed",
                "model_ready": model is not None,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_body()
        if not body:
            self._send_json({"error": "empty request body"}, 400)
            return

        if self.path == "/embed":
            text = body.get("text", "")
            if not text:
                self._send_json({"error": "missing 'text' field"}, 400)
                return
            try:
                t0 = time.time()
                embedding = get_model().encode(text[:512]).tolist()
                elapsed = round((time.time() - t0) * 1000, 1)
                self._send_json({
                    "embedding": embedding,
                    "dim": len(embedding),
                    "time_ms": elapsed,
                })
            except RuntimeError as e:
                self._send_json({"error": str(e), "status": "loading"}, 503)

        elif self.path == "/batch":
            texts = body.get("texts", [])
            if not texts:
                self._send_json({"error": "missing 'texts' field"}, 400)
                return
            try:
                t0 = time.time()
                truncated = [t[:512] for t in texts]
                embeddings = get_model().encode(truncated).tolist()
                elapsed = round((time.time() - t0) * 1000, 1)
                self._send_json({
                    "embeddings": embeddings,
                    "dim": len(embeddings[0]) if embeddings else VEC_DIM,
                    "count": len(embeddings),
                    "time_ms": elapsed,
                })
            except RuntimeError as e:
                self._send_json({"error": str(e), "status": "loading"}, 503)

        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Embedding HTTP API server")
    parser.add_argument("--host", default=os.environ.get("EMBED_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EMBED_PORT", "9001")))
    args = parser.parse_args()

    # Start HTTP server FIRST so healthcheck passes immediately
    server = HTTPServer((args.host, args.port), EmbedHandler)
    print(f"[embed-server] Listening on http://{args.host}:{args.port}", file=sys.stderr, flush=True)

    # Load model in background thread
    import threading
    threading.Thread(target=_background_load_model, daemon=True).start()

    # Also start Unix socket for legacy compatibility
    _start_socket_server()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[embed-server] Shutting down...", file=sys.stderr)
        server.server_close()


def _start_socket_server():
    """Start the Unix socket server in a background thread for backward compatibility."""
    import socket as sock
    import threading

    SOCKET_PATH = "/tmp/embed-server.sock"

    def socket_worker():
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        srv = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
        srv.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)
        srv.listen(8)
        while True:
            client, _ = srv.accept()
            t = threading.Thread(target=_socket_handle, args=(client,), daemon=True)
            t.start()

    t = threading.Thread(target=socket_worker, daemon=True)
    t.start()


def _socket_handle(client):
    """Handle a single Unix socket connection."""
    try:
        data = client.recv(8192)
        if data:
            text = data.decode("utf-8").strip()[:512]
            embedding = get_model().encode(text).tolist()
            response = json.dumps({"embedding": embedding, "dim": len(embedding)})
            client.sendall((response + "\n").encode("utf-8"))
    except Exception as e:
        err = json.dumps({"error": str(e)}) + "\n"
        client.sendall(err.encode("utf-8"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
