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

# Verify uvicorn is installed in the venv
RUN which uvicorn

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

# Copy the virtual environment *directory* from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /home/appuser/app/.venv

# Update PATH to use the virtual environment's Python and scripts/binaries
# The 'bin' directory inside the venv contains executables like uvicorn
ENV PATH="/home/appuser/app/.venv/bin:$PATH"

# Verify the directory structure and uvicorn location (optional, for debugging)
RUN ls -la /home/appuser/app/.venv/bin/ && which uvicorn

# Copy application code
COPY --chown=appuser:appuser . /home/appuser/app

# Ensure the virtual environment's packages are used by default
# This is usually handled by the PATH, but explicitly setting PYTHONPATH can help
ENV PYTHONPATH="${PYTHONPATH}:/home/appuser/app/.venv/lib/python3.9/site-packages"

# Expose the port the app runs on
EXPOSE 7860

# Run the application using the full path if necessary, or rely on PATH
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
# If PATH is correctly set, the above should work. If not, use full path:
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]