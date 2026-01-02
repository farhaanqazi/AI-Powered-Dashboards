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
ARG CACHEBUST=1  # Will change on every rebuild if passed dynamically

# Now copy full source — this layer invalidates on any file change
COPY . .

# Build frontend — guaranteed fresh due to invalidated copy layer
RUN cd frontend && \
    rm -rf dist/ node_modules/.cache && \
    npm run build

# Non-root user
RUN useradd -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]