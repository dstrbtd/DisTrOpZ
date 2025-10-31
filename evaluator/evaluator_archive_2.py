import os
import io
import json
import requests
import hashlib
import bittensor as bt
from exogym.trainer import Trainer
from nanogpt import GPT, GPTConfig, get_dataset
import argparse
import subprocess

# -----------------------------
# 1. Setup basic params
# -----------------------------
DEVICE = "cuda"
SANDBOX_IMAGE = "distropz-sandbox"
DB_URI = os.getenv("DB_URI")  # e.g. postgres://user:pass@db:5432/distropz
MAX_RUNTIME = 900  # seconds


# -----------------------------
# 2. Utility: verify gist integrity
# -----------------------------
def verify_gist(gist_url: str, expected_hash: str):
    """Download gist, verify SHA256, return path if valid."""
    # Get raw gist content
    api_url = gist_url.replace(
        "https://gist.github.com/", "https://api.github.com/gists/"
    )
    resp = requests.get(api_url)
    resp.raise_for_status()
    files = resp.json()["files"]
    if not files:
        raise ValueError("Gist has no files.")
    filename, fileinfo = next(iter(files.items()))
    content = fileinfo["content"]
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch! expected {expected_hash[:8]} got {actual_hash[:8]}"
        )
    path = os.path.join("/tmp", filename)
    with open(path, "w") as f:
        f.write(content)
    return path


# ------------------------------------------------------
# 3. Validate the metrics schema and sanity bounds
# ------------------------------------------------------
def validate_metrics(metrics):
    required = {"throughput": int, "loss": (int, float), "comm": int}
    for key, expected_type in required.items():
        if key not in metrics:
            raise ValueError(f"Missing key: {key}")
        if not isinstance(metrics[key], expected_type):
            raise ValueError(f"{key} has wrong type: {type(metrics[key])}")
    # bounds
    if not (0 <= metrics["throughput"] < 10**9):
        raise ValueError("throughput out of range")
    if not (0.0 <= metrics["loss"] < 1e6):
        raise ValueError("loss out of range")
    if not (0 <= metrics["comm"] < 10**12):
        raise ValueError("comm out of range")
    return metrics


# ------------------------------------------------------
# 4. Execute untrusted miner gist inside sandbox container
# ------------------------------------------------------
def run_in_sandbox(gist_path):
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--cpus=2",
        "--memory=4g",
        "-v",
        f"{gist_path}:/sandbox/strategy.py:ro",
        SANDBOX_IMAGE,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_RUNTIME)
    except subprocess.TimeoutExpired:
        raise TimeoutError("Sandbox timed out")

    if proc.returncode != 0:
        raise RuntimeError(f"Sandbox failed: {proc.stderr}")

    try:
        metrics = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON output: {proc.stdout[:200]}")

    if "error" in metrics:
        raise RuntimeError(f"Sandbox error: {metrics['error']}")
    return validate_metrics(metrics)


# -----------------------------
# 3. Main validation loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument(
        "--number_of_nodes",
        type=int,
        default=2,
        help="Nuber of GPUs to use for testing",
    )
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Maximum number of steps to test"
    )
    parser.add_argument("--mode_size", help="NanoGPT model size")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--netuid", type=int, default=178, help="Bittensor network UID."
    )

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"

    bt.logging.setLevel("INFO")
    bt.logging.info(config)
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    os.makedirs(config.output_dir, exist_ok=True)

    bt.logging.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")

    # Load dataset once
    train_dataset, vocab_size = get_dataset(
        "shakespeare",
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * config.number_of_nodes,
    )
    val_dataset, vocab_size = get_dataset(
        "shakespeare", block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )

    metrics = {}

    for uid, hotkey in enumerate(metagraph.hotkeys):
        bt.logging.info(f"🔍 Checking miner UID={uid} hotkey={hotkey}")

        try:
            meta = subtensor.get_metadata(
                wallet, uid=uid, netuid=config.netuid, key="strategy_gist"
            )
            if not meta:
                bt.logging.warning(f"❌ No strategy_gist for {hotkey}")
                continue

            data = json.loads(meta)
            gist_url, sha = data["url"], data["sha256"]
            bt.logging.info(f"Found gist: {gist_url}")

            # verify and save
            path = verify_gist(gist_url, sha)

            # dynamically import strategy
            namespace = {}
            exec(open(path).read(), namespace)
            if "STRATEGY" not in namespace:
                bt.logging.warning(f"❌ No STRATEGY object found in {gist_url}")
                continue
            strategy = namespace["STRATEGY"]

            # create model + trainer
            model = GPT(GPTConfig.gpt2_size_map(config.model_size))
            trainer = Trainer(model, train_dataset, val_dataset, device=DEVICE)

            # train & benchmark
            _, metrics_out = trainer.fit(
                max_steps=config.max_steps,
                strategy=strategy,
                num_epochs=1,
                num_nodes=config.number_of_nodes,
                device=DEVICE,
                batch_size=16,
                minibatch_size=2,
                shuffle=False,
                val_size=128,
                val_interval=10,
                run_name=f"uid{uid}_{hotkey[:6]}",
            )

            metrics[hotkey] = metrics_out
            bt.logging.success(f"✅ Finished miner {hotkey}: {metrics_out}")

        except Exception as e:
            bt.logging.error(f"⚠️ Error validating {hotkey}: {e}")

    # save results
    out_path = os.path.join(
        config.output_dir,
        f"metrics-gpt-{config.model_size}-{config.number_of_nodes}-{config.max_steps}.json",
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    bt.logging.success(f"Saved metrics → {out_path}")


if __name__ == "__main__":
    main()
