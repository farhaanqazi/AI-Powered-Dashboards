# --- Single Stage Build ---
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

# Copy application code
COPY . .

# Install frontend dependencies and build the React app
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Switch back to main directory
WORKDIR /app

# Create a non-root user and switch to it
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 7860

# Run the application using python -m uvicorn to ensure it's found
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]