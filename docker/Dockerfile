# docker/Dockerfile
# 1. Base Image: Use a slim version of Python for a smaller attack surface and image size
FROM python:3.11-slim

# 2. Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures logs are sent straight to the terminal (important for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Set the working directory
WORKDIR /app

# 4. Install system dependencies (if any are needed, e.g., gcc for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# We copy requirements.txt first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the project structure
# We copy src, configs, and scripts. We exclude notebooks and data/ (data should be mounted/external)
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# 7. Define the entry point
# This allows us to run the container by simply naming the script
ENTRYPOINT ["python", "-m", "scripts.train"]
