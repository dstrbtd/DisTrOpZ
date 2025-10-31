import argparse, os, json, subprocess, hashlib, tempfile, traceback, bittensor as bt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import requests
import re

# --- Config ---
SANDBOX_IMAGE = "distropz-sandbox"
DB_URI = os.getenv("DB_URI")  # e.g. postgres://user:pass@db:5432/distropz
MAX_RUNTIME = 900  # seconds


# ------------------------------------------------------
# 1. Validate the metrics schema and sanity bounds
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
# 2. Execute untrusted miner gist inside sandbox container
# ------------------------------------------------------
def run_in_sandbox(gist_path):
    # Install docker on something other than runpod
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


# ------------------------------------------------------
# 3. Insert verified metrics into DB (trusted context)
# ------------------------------------------------------
def log_to_db(hotkey, metrics):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO validator_scores (hotkey, throughput, loss, comm)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (hotkey) DO UPDATE
        SET throughput = EXCLUDED.throughput,
            loss = EXCLUDED.loss,
            comm = EXCLUDED.comm;
        """,
        (hotkey, metrics["throughput"], metrics["loss"], metrics["comm"]),
    )
    conn.commit()
    cur.close()
    conn.close()


# -------------------------------------------------------------------------------
# 4. Validate a single miner: fetch gist, verify hash, run sandbox, log results
# -------------------------------------------------------------------------------
def validate_miner(hotkey, gist_url, expected_hash):
    """Fetch gist, verify hash, run sandbox, log results"""
    bt.logging.info(f"Evaluating miner {hotkey}")

    # --- fetch gist ---
    match = re.search(r"([0-9a-fA-F]{8,})$", gist_url)
    if not match:
        raise ValueError(f"Could not parse gist ID from URL: {gist_url}")
    gist_id = match.group(1)

    api_url = f"https://api.github.com/gists/{gist_id}"
    resp = requests.get(api_url)
    if resp.status_code == 404:
        raise ValueError(f"Gist not found or not public: {gist_url}")
    resp.raise_for_status()

    files = resp.json().get("files", {})
    if not files:
        raise ValueError("Gist has no files.")

    filename, fileinfo = next(iter(files.items()))
    code = fileinfo["content"]

    # Verify hash
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch! expected {expected_hash[:8]}, got {actual_hash[:8]}"
        )

    # --- write to temp file ---
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        metrics = run_in_sandbox(tmp_path)
        # log_to_db(hotkey, metrics)
        bt.logging.success(f"✅ Miner {hotkey}: {metrics}")
        return metrics
    except Exception as e:
        bt.logging.error(f"❌ Failed miner {hotkey}: {e}")
        traceback.print_exc()
        return {}
    finally:
        os.remove(tmp_path)


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
    parser.add_argument("--mode_size", default="tiny", help="NanoGPT model size")
    parser.add_argument("--dataset", default="shakespeare", help="Dataset name")
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

    metrics = {}

    for uid, hotkey in enumerate(metagraph.hotkeys):
        uid = 3
        hotkey = [metagraph.hotkeys[3]]
        bt.logging.info(f"🔍 Checking miner UID={uid} hotkey={hotkey}")

        try:
            meta = subtensor.get_commitment(netuid=config.netuid, uid=uid)
            if not meta:
                bt.logging.warning(f"❌ No strategy_gist for {hotkey}")
                continue

            gist_url = meta[64:]
            sha = meta[:64]
            bt.logging.info(f"Found gist: {gist_url}")

            # verify and save
            hotkey_metrics = validate_miner(hotkey, gist_url, sha)

            metrics[hotkey] = hotkey_metrics
            bt.logging.success(f"✅ Finished miner {hotkey}: {hotkey_metrics}")

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
