FROM astral/uv:python3.11-bookworm-slim

WORKDIR /app

# System dependencies:
# - libvips-dev: required by pyvips for WSI I/O
# - libgl1, libglib2.0-0: cellpose/opencv runtime deps
# - libopenslide0: OpenSlide-backed WSI formats
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    procps \
    libvips-dev \
    libgl1 \
    libglib2.0-0 \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY README.md ./

# Install dependencies
RUN uv sync --no-dev

# Add venv to PATH so CLI commands are available
ENV PATH="/app/.venv/bin:$PATH"

# Set entrypoint
CMD ["bash"]
