# --- Builder Stage ---
FROM python:3.9-slim AS builder

# Install system dependencies needed for some Python packages (e.g., pandas, numpy compilation)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for building (optional, but good practice if build tools need specific permissions)
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Create a virtual environment
ENV VENV_PATH=/app/.venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt requirements.txt

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Runtime Stage ---
FROM python:3.9-slim AS runtime

# Create the same non-root user for runtime
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Switch to the non-root user before setting up the filesystem
USER appuser
# Set home directory for the user
ENV HOME=/home/appuser
# Set the working directory inside the user's home
WORKDIR $HOME/app

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /home/appuser/app/.venv

# Update PATH to use the virtual environment's Python and pip
ENV PATH="/home/appuser/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . /home/appuser/app

# Ensure the virtual environment's packages are used by default
ENV PYTHONPATH="${PYTHONPATH}:/home/appuser/app"

# Expose the port the app runs on
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]