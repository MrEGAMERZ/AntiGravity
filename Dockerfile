# ───────────────────────────────────────────────────────────
# AntiGravity v4 — Dockerfile for Hugging Face Spaces (Docker SDK)
# Runs the FastAPI server on port 7860
# Cache-bust: 20260329
# ───────────────────────────────────────────────────────────

FROM python:3.11-slim

# HF Spaces requires a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Copy and install dependencies first (layer caching)
COPY --chown=user server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY --chown=user . .

# Expose the HF Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
