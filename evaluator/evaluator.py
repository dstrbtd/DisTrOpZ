import argparse, os, json, subprocess, hashlib, tempfile, traceback, bittensor as bt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import requests
import datetime
import re

# --- Config ---
SANDBOX_IMAGE = "distropz_sandbox"
INFLUXDB_MEASUREMENT = "mechanism1_metrics"
MAX_RUNTIME = 900  # seconds


# ------------------------------------------------------
# 1. Validate the metrics schema and sanity bounds
# ------------------------------------------------------
def validate_metrics(metrics):
    required = {"throughput": int, "loss": (int, float), "communication": int}
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
    if not (0 <= metrics["communication"] < 10**12):
        raise ValueError("communication out of range")
    return metrics


# ------------------------------------------------------
# 2. Execute untrusted miner gist inside sandbox container
# ------------------------------------------------------
def run_in_sandbox(gist_path, config):
    # # Install docker on something other than runpod
    # cmd = [
    #     "docker",
    #     "run",
    #     "--rm",
    #     "--gpus",
    #     "all",
    #     "--network=none",
    #     # "--cpus=2",
    #     # "--memory=4g",
    #     "--shm-size=8g",
    #     # environment variables (each one must be a separate -e)
    #     "-e",
    #     f"NUM_NODES={config.number_of_nodes}",
    #     "-e",
    #     f"MAX_STEPS={config.max_steps}",
    #     "-e",
    #     f"DATASET={config.dataset}",
    #     "-e",
    #     f"MODEL_SIZE={config.model_size}",
    #     # volume mount
    #     "-v",
    #     f"{gist_path}:/app/sandbox/strategy.py:ro",
    #     # image name
    #     SANDBOX_IMAGE,
    # ]
    print(gist_path)
    cmd = ["/root/.dto/bin/python", "/root/DisTrOpZ/evaluator/evaluation_sandbox.py"]
    # cmd = [f"NUM_NODES={config.number_of_nodes}", f"MAX_STEPS={config.max_steps}",f"DATASET={config.dataset}", f"MODEL_SIZE={config.model_size}", "/root/.dto/bin/python", "/root/DisTrOpZ/evaluator/evaluation_sandbox.py"]
    # cmd = f"NUM_NODES={config.number_of_nodes} MAX_STEPS={config.max_steps} DATASET={config.dataset} MODEL_SIZE={config.model_size} /root/.dto/bin/python /root/DisTrOpZ/evaluator/evaluation_sandbox.py"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        universal_newlines=True,
        # env=env,
        env={
            "NUM_NODES": str(config.number_of_nodes),
            "MAX_STEPS": str(config.max_steps),
            "DATASET": config.dataset,
            "MODEL_SIZE": config.model_size,
        },
    )

    full_output = []
    try:
        for line in process.stdout:
            print(line, end="")  # stream to your terminal in real-time
            full_output.append(line)
        process.wait(timeout=None)
    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutError("Sandbox timed out")

    if process.returncode != 0:
        tail = "".join(full_output[-50:])  # last lines for context
        raise RuntimeError(f"Sandbox failed (rc={process.returncode}):\n{tail}")

    # parse only the final JSON line
    # metrics = json.loads(full_output[-1].strip() if full_output else "")
    metrics = json.loads(
        "{" + "".join(full_output[-50:]).split("{")[-1].strip() if full_output else ""
    )

    if "error" in metrics:
        raise RuntimeError(f"Sandbox error: {metrics['error']}")

    return validate_metrics(metrics)


# ------------------------------------------------------
# 3. Insert verified metrics into DB (trusted context)
# ------------------------------------------------------
def log_to_db(write_api, hotkey, uid, metrics, config, gist_url, current_block):
    point = (
        Point(INFLUXDB_MEASUREMENT)
        .tag("number_of_nodes", config.number_of_nodes)
        .tag("max_steps", config.max_steps)
        .tag("model_size", config.model_size)
        .tag("dataset", config.dataset)
        .tag("current_block", current_block)
        .tag("hotkey", hotkey)
        .tag("uid", uid)
        .tag("gist_url", gist_url)
        .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)
    )
    for k, v in metrics.items():
        point.field(k, v)
    write_api.write(
        bucket=config.influxdb.bucket, org=config.influxdb.org, record=point
    )


# -------------------------------------------------------------------------------
# 4. Validate a single miner: fetch gist, verify hash, run sandbox, log results
# -------------------------------------------------------------------------------
def validate_miner(
    write_api, hotkey, uid, gist_url, expected_hash, config, current_block
):
    """Fetch gist, verify hash, run sandbox, log results"""
    bt.logging.info(f"Evaluating miner {hotkey}")

    # --- fetch gist ---
    match = re.search(r"([0-9a-fA-F]{8,})$", gist_url)
    if not match:
        raise ValueError(f"Could not parse gist ID from URL: {gist_url}")
    gist_id = match.group(1)

    api_url = f"https://api.github.com/gists/{gist_id}"
    resp = requests.get(api_url)
    # if resp.status_code == 404:
    #     raise ValueError(f"Gist not found or not public: {gist_url}")
    # resp.raise_for_status()

    # files = resp.json().get("files", {})
    # if not files:
    #     raise ValueError("Gist has no files.")

    # filename, fileinfo = next(iter(files.items()))
    # code = fileinfo["content"]

    # # Verify hash
    # actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    # if actual_hash != expected_hash:
    #     raise ValueError(
    #         f"Hash mismatch! expected {expected_hash[:8]}, got {actual_hash[:8]}"
    #     )

    # # --- write to temp file ---
    # with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
    #     tmp.write(code)
    #     tmp_path = tmp.name

    # if hotkey != "5EEqeZe2KmWTHKRr48xZNgfDXZCJScfTMvt2daoMxKz1Zifw":
    #     raise ValueError("Gist has no files.")
    if hotkey == "5EvvqR8EJhQYVyk6avp2dpkLymR95StUqPoRSSN7sD9FUSWj":
        path = "/root/DisTrOpZ/miner/miner_diloco.py"
    elif hotkey == "5EEqeZe2KmWTHKRr48xZNgfDXZCJScfTMvt2daoMxKz1Zifw":
        path = "/root/DisTrOpZ/miner/miner_sparseloco.py"
    elif hotkey == "5HW6iTCNfk9xRmNbFv7PKGpJL99JU2wzco4ABJxywKZGgjJA":
        path = "/root/DisTrOpZ/miner/miner_demo.py"
    elif hotkey == "5EvFbREcHj3gC9tRCbQ5E4CF25UCAVsJj4pFyzFqHrbgn9Rg":
        path = "/root/DisTrOpZ/miner/miner_federated_averaging.py"

    # path = "/sandbox/sandbox.py"
    tmp_path = path
    import shutil

    # Copy the file
    shutil.copy(path, "/root/DisTrOpZ/evaluator/sandbox/strategy.py")

    try:
        bt.logging.success(f"✅ Run {path} in sandbox")
        metrics = run_in_sandbox(tmp_path, config)
        # log_to_db(write_api, hotkey, uid, metrics, config, gist_url, current_block)
        bt.logging.success(f"✅ Miner {hotkey}: {metrics}")
        return metrics
    except Exception as e:
        bt.logging.error(f"❌ Failed miner {hotkey}: {e}")
        traceback.print_exc()
        return {}
    # finally:
    #     os.remove(tmp_path)


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
    parser.add_argument("--model_size", default="large", help="NanoGPT model size")
    parser.add_argument("--dataset", default="shakespeare", help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--netuid", type=int, default=178, help="Bittensor network UID."
    )
    parser.add_argument("--influxdb.measurement", default="distropz_metrics")
    parser.add_argument("--influxdb.bucket", required=True)
    parser.add_argument("--influxdb.org", required=True)
    parser.add_argument("--influxdb.url", default="http://localhost:8086")
    parser.add_argument(
        "--influxdb.token", help="InfluxDB token (or set INFLUXDB_TOKEN)"
    )

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"

    influx = InfluxDBClient(
        url=config.influxdb.url, token=config.influxdb.token, org=config.influxdb.org
    )
    write_api = influx.write_api()
    read_api = influx.query_api()

    bt.logging.setLevel("INFO")

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    current_block = subtensor.block

    os.makedirs(config.output_dir, exist_ok=True)

    bt.logging.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(
        config.output_dir,
        f"metrics-gpt-{config.model_size}-{config.dataset}-{config.number_of_nodes}-{config.max_steps}-{datetime_stamp}.json",
    )

    metrics = {}
    urls = {}

    for uid, hotkey in enumerate(metagraph.hotkeys):
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
            hotkey_metrics = validate_miner(
                write_api, hotkey, uid, gist_url, sha, config, current_block
            )

            urls[hotkey] = gist_url
            metrics[hotkey] = hotkey_metrics
            bt.logging.success(f"✅ Finished miner {hotkey}: {hotkey_metrics}")

            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            bt.logging.error(f"⚠️ Error validating {hotkey}: {e}")

    df = pd.DataFrame(metrics).T

    metric_names = ["throughput", "loss", "communication"]

    # min-max normalize each metric
    norm = (df[metric_names] - df[metric_names].min()) / (
        df[metric_names].max() - df[metric_names].min()
    )

    df["throughput_norm"] = norm["throughput"]
    df["loss_norm"] = 1 - norm["loss"]  # lower loss is better
    df["communication_norm"] = (
        1 - norm["communication"]
    )  # lower communication is better

    df["score"] = (
        (1 / 3) * df["throughput_norm"]
        + (1 / 3) * df["loss_norm"]
        + (1 / 3) * df["communication_norm"]
    )

    for hotkey in metrics.keys():
        if hotkey in urls:
            gist_url = urls[hotkey]
            uid = metagraph.hotkeys.index(
                "5HdXxbuzWdzw13eamutFwGXX57ycisFxivXzP5ZLtF1FTfF5"
            )
            metrics[hotkey]["score"] = df.loc[hotkey, "score"].item()
            log_to_db(
                write_api, hotkey, uid, metrics[hotkey], config, gist_url, current_block
            )

    # save results
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    bt.logging.success(f"Saved metrics → {out_path} → {current_block}")


if __name__ == "__main__":
    main()
