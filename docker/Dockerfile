# docker/Dockerfile
# Base Image: Use a slim version of Python for a smaller attack surface and image size
FROM python:3.11-slim

# 1. Define the environment variables explicitly
# We MUST define APP_HOME so that WORKDIR knows where to go
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory to /app
# This ensures pathlib.Path(__file__).resolve().parents[2] resolves to /app
WORKDIR $APP_HOME

# 3. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the project structure
# We do NOT copy data/ here because the volume mount in docker-compose 
# will provide the data at runtime.
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# 5. Entrypoint
ENTRYPOINT ["python", "-m", "scripts.train"]

