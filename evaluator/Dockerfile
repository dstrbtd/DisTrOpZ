# =====================================================
# DisTrOpZ Miner Sandbox
# Runs untrusted miner code in a locked-down environment
# =====================================================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
# FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Install only minimal deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip git tini libgomp1 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# --- Environment setup ---
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

WORKDIR /app

# 1) copy project metadata first (better layer caching for deps)
COPY pyproject.toml README.md ./

# 2) copy source needed for the package
COPY exogym ./exogym
COPY evaluator/nanogpt ./nanogpt
COPY evaluator/evaluation_sandbox.py ./evaluation_sandbox.py

# --- Copy the pre-cache helper script ---
COPY evaluator/precache_dataset.py ./precache_dataset.py

# 3) Install PyTorch with CUDA first, then editable install
RUN python3 -m pip install --upgrade pip && \
    # python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir -e .

# # --- Prepare data directory (dataset will be mounted at runtime) ---
# RUN mkdir -p /app/data && chown -R 1001:1001 /app
# # NOTE: Mount the dataset at runtime with: -v /path/to/owt-data:/app/data

RUN mkdir -p /sandbox && chmod 755 /sandbox

# --- Create unprivileged user safely ---
RUN id -u sandboxuser 2>/dev/null || useradd -m -u 1001 sandboxuser

# --- Create data and logs directories with proper permissions ---
RUN mkdir -p /app/data /app/logs && chown -R sandboxuser:sandboxuser /app/data /app/logs

# --- Pre-cache the dataset during build (needs network) ---
RUN python3 precache_dataset.py && chown -R sandboxuser:sandboxuser /app/data

# --- Switch to unprivileged user ---
USER sandboxuser
WORKDIR /app

# --- Set entrypoint ---
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "evaluation_sandbox.py"]
