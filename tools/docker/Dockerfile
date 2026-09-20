# Development image for Open Badger's initial model/PyTorch workflow.

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ARG LEROBOT_COMMIT=a4f15bf347dee7eb8a8c5f4a70a37f476b091113
ENV LEROBOT_ROOT=/opt/lerobot

WORKDIR /workspace/open-badger

# Install only operating-system tools needed for Python development and pulling
# upstream model/data artifacts. GPU drivers remain provided by the host.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Keep project packages out of Ubuntu's externally managed system Python.
RUN python3 -m venv "$VIRTUAL_ENV"

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install the upstream LeRobot MolmoAct2 policy at a pinned commit. Open
# Badger adapts to this policy; it does not duplicate its model implementation.
RUN git clone https://github.com/allenai/lerobot.git "$LEROBOT_ROOT" \
    && git -C "$LEROBOT_ROOT" checkout "$LEROBOT_COMMIT"

# Install Open Badger and all current development/training extras. The extra
# PyTorch index supplies the CUDA-compatible wheels declared in pyproject.toml.
RUN python3 -m pip install --upgrade pip setuptools \
    && python3 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e ".[dev,training,model,data,lora,logging]" \
    && python3 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e "$LEROBOT_ROOT[molmoact2]"

CMD ["bash"]
