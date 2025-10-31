# =====================================================
# DisTrOpZ Miner Sandbox
# Runs untrusted miner code in a locked-down environment
# =====================================================

FROM python:3.10-slim

# --- Create an unprivileged user ---
RUN useradd -m -u 1001 sandboxuser

# --- Install only minimal deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 tini && \
    rm -rf /var/lib/apt/lists/*

# --- Switch to unprivileged user ---
USER sandboxuser
WORKDIR /home/sandboxuser

# --- Copy runtime code ---
# (The validator will COPY or mount miner's gist script here at runtime)
COPY evaluation_sandbox.py evaluation_sandbox.py

# --- Set entrypoint ---
ENTRYPOINT ["/usr/bin/tini", "--", "python", "evaluation_sandbox.py"]
