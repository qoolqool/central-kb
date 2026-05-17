# Central KB

Centralized multi-project knowledge base with simhash dedup, drift detection, and global pattern promotion.
A shared embedding server (bge-large-en-v1.5, 1024-dim) serves all connected containers — no per-container model downloads needed.

## Services

| Service | Port | Description |
|---------|------|-------------|
| `central-kb` | 9000 | FastAPI + SQLite + FTS5 hybrid search |
| `embed-server` | 9001 | sentence-transformers, bge-large, CPU-only PyTorch |

## Quick Start

> **Proxied environment?** If your network uses SSL inspection (corporate proxy),
> you may need to trust your organization's CA certificate for pip, curl, and
> Python to reach PyPI and HuggingFace. See `Dockerfile.embed-server` for instructions.

```bash
# Build and start (first build takes ~5min for PyTorch)
docker compose up -d

# Wait for both services to become healthy
docker compose ps

# Check health
curl http://localhost:9000/health
curl http://localhost:9001/health

# Seed from an existing local KB
python3 scripts/seed.py --project my-project --from /path/to/agentdb.sqlite3
```

## CLI Usage

Install the `kb` CLI:

```bash
pip install -e ".[dev]"
export CENTRAL_KB_URL=http://localhost:9000
export CENTRAL_KB_PROJECT=my-project
```

| Command | Description |
|---------|-------------|
| `kb submit --project my-project` | Submit local KB to central |
| `kb pull --project my-project` | Pull project entries |
| `kb pull --project my-project --global` | Include global namespace |
| `kb search "query" --scope my-project` | Hybrid search |
| `kb drift --project my-project` | Show drift report |
| `kb candidates` | List promotion candidates |
| `kb promote 7 approve` | Approve promotion |
| `kb conflicts` | List pending conflicts |
| `kb resolve 42 --resolution keep_existing` | Resolve a conflict |

## Embedding Server

The embed-server uses CPU-only PyTorch (from `https://download.pytorch.org/whl/cpu`) to avoid
hash mismatch issues with nvidia-* GPU packages. It provides:

- `POST /embed` — single text → 1024-dim vector
- `POST /batch` — multiple texts → multiple vectors
- `GET /health` — health check with model loading status

Other containers (e.g., playground tooling containers) can use the embed server by setting:

```bash
export EMBED_SERVER_URL=http://embed-server:9001
```

Or via `host.containers.internal:9001` when not on the same Docker network.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/submit` | 3-phase ingest (dedup → conflict → publish) |
| `GET` | `/pull` | Cursor-based pull of accepted entries |
| `GET` | `/search` | Hybrid search (cosine + FTS5 blended) |
| `GET` | `/drift` | Cross-project drift detection |
| `GET` | `/candidates` | Promotion candidates |
| `POST` | `/promote` | Approve/reject a candidate |
| `GET` | `/conflicts` | Pending conflicts |
| `POST` | `/conflicts/{id}/resolve` | Resolve a conflict |
| `POST` | `/reset` | Clear all data and reinitialize schema |

## Development

```bash
# Install deps (editable mode)
pip install -e ".[dev]"

# Run tests
python3 -m pytest tests/ -v

# Run dev server with live reload
bash scripts/run_dev.sh
```

## Integration with Playground

This repo is designed to run alongside [playground](https://github.com/qoolqool/playground) containers.
Playground tooling containers auto-detect the central-kb services via `host.containers.internal`
and use them for shared knowledge. When central-kb is not available, they fall back to local
Ollama embeddings and local SQLite.

```bash
# Clone both repos side by side
git clone https://github.com/qoolqool/playground
git clone https://github.com/qoolqool/central-kb

# Start central-kb
cd central-kb && docker compose up -d

# Start playground (will find central-kb automatically)
cd ../playground && ./start.sh
```

## Architecture

```
embed-server (port 9001) ──HTTP──▶ central-kb (port 9000)
  sentence-transformers            FastAPI + SQLite + FTS5
  bge-large-en-v1.5                Simhash dedup
  CPU-only PyTorch                 Hybrid search (cosine + BM25)
                                   Drift detection
                                   Global pattern promotion

▲        ▲        ▲
│        │        │
▼        ▼        ▼
proj-A   proj-B   proj-C
(playground containers)
```
