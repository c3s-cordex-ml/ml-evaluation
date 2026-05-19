FROM python:3.13-slim

# Install system dependencies (curl and git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Ensure pixi is in PATH
ENV PATH="/root/.pixi/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install environment and dependencies
RUN pixi install --locked -e dev

# Define entrypoint to enable direct CLI execution in Kubernetes
ENTRYPOINT ["pixi", "run", "-e", "dev", "ml-eval"]
