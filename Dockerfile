# --- Cache-Busted Single Stage Build ---
FROM python:3.9-slim

# System deps + Node
RUN apt-get update && apt-get install -y gcc g++ curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Frontend deps (cacheable if package files unchanged)
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci  # Use ci for deterministic install

# Critical: Dynamic cache buster BEFORE copying source
ARG CACHEBUST=1  # IMPORTANT: Pass --build-arg CACHEBUST=$(date +%s) for cache invalidation.

# Clerk publishable key — Vite inlines VITE_* at build time, so it MUST be
# present during `npm run build`. On Hugging Face Spaces, add this as a Space
# secret (Settings → Variables and secrets) and it is exposed as a build ARG.
ARG VITE_CLERK_PUBLISHABLE_KEY
ENV VITE_CLERK_PUBLISHABLE_KEY=$VITE_CLERK_PUBLISHABLE_KEY

# Now copy full source — this layer invalidates on any file change
COPY . .

# Build frontend — guaranteed fresh due to invalidated copy layer
RUN cd frontend && \
    rm -rf dist/ node_modules/.cache && \
    npm run build

# Entrypoint runs the API, plus the Arq worker iff JOB_QUEUE_ENABLED=true.
# Default behaviour (flag unset) is identical to the previous single CMD —
# Hugging Face is unaffected unless the flag is explicitly set.
RUN chmod +x docker/entrypoint.sh

# Non-root user
RUN useradd -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "/app/docker/entrypoint.sh"]