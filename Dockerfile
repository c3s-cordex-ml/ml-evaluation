FROM ghcr.io/prefix-dev/pixi:latest

# Set work directory
WORKDIR /workdir

# Copy application files
COPY . .

USER root
RUN apt-get update -y && \
    apt-get install -y g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies and set up environment
RUN pixi install && pixi clean

# Install the package
RUN pixi run python -m pip install --no-cache-dir --no-deps .

# Use pixi run as default entrypoint
ENTRYPOINT ["pixi", "run"]
