FROM python:3.11-slim

WORKDIR /app

# --- CA Certificates (for corporate proxy / Cloudflare Gateway) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Cloudflare Gateway CA if available in build context
COPY cloudflare-gateway.cr[t] /tmp/
RUN if [ -f /tmp/cloudflare-gateway.crt ]; then \
      cp /tmp/cloudflare-gateway.crt /usr/local/share/ca-certificates/ && \
      update-ca-certificates --fresh; \
    else \
      echo "⚠ No cloudflare-gateway.crt - using system CAs only"; \
    fi

# Set SSL environment variables for pip, curl, and Python
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install Python dependencies
COPY pyproject.toml .
# sentence-transformers NOT needed here — embedding is delegated to embed-server sidecar
# Avoids hash mismatch errors from nvidia-* GPU packages (known pip issue)
# See: https://github.com/huggingface/sentence-transformers/issues/1409
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx

# Copy application code
COPY app/ app/
COPY kb_cli/ kb_cli/

RUN mkdir -p /data

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
