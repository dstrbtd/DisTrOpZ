# =====================================================
# DisTrOpZ Miner Sandbox
# Runs untrusted miner code in a locked-down environment
# =====================================================

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

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

# 3) editable install
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -e .

# --- Pre-cache the dataset ---
RUN mkdir -p /app/data && \
    python3 precache_dataset.py && \
    chown -R 1001:1001 /app

RUN mkdir -p /sandbox && chmod 755 /sandbox

# --- Create unprivileged user safely ---
RUN id -u sandboxuser 2>/dev/null || useradd -m -u 1001 sandboxuser

# --- Switch to unprivileged user ---
USER sandboxuser
WORKDIR /app

# --- Set entrypoint ---
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "evaluation_sandbox.py"]
