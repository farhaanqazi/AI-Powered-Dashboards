# --- Single Stage Build with Cache Busting ---
FROM python:3.9-slim

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt requirements.txt

# Install Python dependencies globally (not in user directory) to ensure executables are in PATH
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Optimized Frontend Build with Cache Busting ---
# Copy frontend dependency manifests
COPY frontend/package.json frontend/package-lock.json ./frontend/

# Install frontend dependencies first to leverage Docker cache
RUN cd frontend && npm install

# Add cache-busting timestamp to ensure fresh builds
RUN echo "CACHE_BUST_TIMESTAMP=$(date +%s)" > /tmp/cache_bust.txt

# Copy the rest of the application code
COPY . .

# Build the React frontend with cache-busting
RUN cd frontend && \
    # Clear any existing build cache to ensure fresh build
    rm -rf dist/ && \
    npm run build && \
    echo "✅ Build command completed at $(date)" && \
    pwd && \
    ls -la && \
    ls -la dist/ && \
    echo "✅ Frontend built successfully with cache-busting"

# Create a non-root user and switch to it
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 7860

# Run the application using python -m uvicorn to ensure it's found
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]