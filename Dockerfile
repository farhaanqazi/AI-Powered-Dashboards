# --- Single Stage Build ---
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

# Switch to the non-root user
USER appuser

# Set home directory for the user
ENV HOME=/home/appuser

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt requirements.txt

# Install dependencies directly into the system site-packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]