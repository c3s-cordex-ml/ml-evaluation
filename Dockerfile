FROM ghcr.io/prefix-dev/pixi:latest

# Set work directory
WORKDIR /workdir

# Copy application files
COPY . .

USER root
RUN apt-get update -y && \
    apt-get install -y g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies and the package itself (editable, via [pypi-dependencies] in pixi.toml)
RUN pixi install && pixi clean cache --yes

# Use pixi run as default entrypoint
ENTRYPOINT ["pixi", "run"]
