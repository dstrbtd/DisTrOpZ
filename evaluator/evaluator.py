import argparse, os, json, subprocess, hashlib, tempfile, traceback, bittensor as bt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import requests
import datetime
import re
import shutil
import gc
import torch
import time
import logging
from dotenv import load_dotenv
from logger import setup_loki_logging, add_sandbox_handler

load_dotenv()

# Config
SANDBOX_IMAGE = "distropz_sandbox"
WHITELISTED_HOTKEYS = [
    "5ECDEtiHDP7tXeG3L7PViHsjSUPCsijKEokrFWhdXuATDjH1",
    "5EvvqR8EJhQYVyk6avp2dpkLymR95StUqPoRSSN7sD9FUSWj",
    "5HCDdzGFn2FN5bTJCGrXREWQFMCspDYcT9QeETrzkwvkDzMT",
    "5EEqeZe2KmWTHKRr48xZNgfDXZCJScfTMvt2daoMxKz1Zifw",
    "5HW6iTCNfk9xRmNbFv7PKGpJL99JU2wzco4ABJxywKZGgjJA",
    "5EvFbREcHj3gC9tRCbQ5E4CF25UCAVsJj4pFyzFqHrbgn9Rg",
]


# Validate the metrics schema and confirm each metric is within bounds
def validate_metrics(metrics):
    # Metric Schema
    required = {"throughput": int, "loss": (int, float), "communication": int}
    for key, expected_type in required.items():
        if key not in metrics:
            raise ValueError(f"Missing key: {key}")
        if not isinstance(metrics[key], expected_type):
            raise ValueError(f"{key} has wrong type: {type(metrics[key])}")

    # Metric bounds
    if not (0 <= metrics["throughput"] < 10**9):
        raise ValueError("throughput out of range")
    if not (0.0 <= metrics["loss"] < 1e6):
        raise ValueError("loss out of range")
    if not (0 <= metrics["communication"] < 10**12):
        raise ValueError("communication out of range")
    return metrics


# Execute untrusted miner gist inside sandbox container
def run_in_sandbox(gist_path, config, sandbox_logger=None):
    # Install docker on something other than runpod
    # 2.5 hour timeout (9000 seconds)
    cmd = [
        "timeout",
        "--signal=KILL",
        "9000",
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network=none",
        # "--cpus=2",
        # "--memory=4g",
        "--shm-size=64g",
        "--user",
        "root",
        # environment variables (each one must be a separate -e)
        "-e",
        f"NUM_NODES={config.number_of_nodes}",
        "-e",
        f"MAX_STEPS={config.max_steps}",
        "-e",
        f"DATASET={config.dataset}",
        "-e",
        f"MODEL_SIZE={config.model_size}",
        "-e",
        "NCCL_P2P_DISABLE=1",
        "-e",
        "NCCL_IB_DISABLE=1",
        "-e",
        "NCCL_SOCKET_IFNAME=lo",
        # volume mounts
        "-v",
        f"{gist_path}:/sandbox/strategy.py:ro",
        # "-v",
        # "/root/DisTrOpZ/results:/app/results",
        # image name
        SANDBOX_IMAGE,
    ]
    bt.logging.info(f"Running sandbox with strategy: {gist_path}")
    # cmd = ["/root/.venv/bin/python", "/root/DisTrOpZ/evaluator/evaluation_sandbox.py"]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        universal_newlines=True,
        env={
            "NUM_NODES": str(config.number_of_nodes),
            "MAX_STEPS": str(config.max_steps),
            "DATASET": config.dataset,
            "MODEL_SIZE": config.model_size,
            "NCCL_P2P_DISABLE": str(1),
        },
    )

    full_output = []
    try:
        for line in process.stdout:
            line_stripped = line.rstrip()
            print(line, end="")  # stream to your terminal in real-time
            full_output.append(line)
            # Forward sandbox output to Loki if logger is available
            if sandbox_logger and line_stripped:
                sandbox_logger.info(line_stripped)
        process.wait(timeout=None)
    except subprocess.TimeoutExpired:
        process.kill()
        raise TimeoutError("Sandbox timed out")

    if process.returncode != 0:
        tail = "".join(full_output[-50:])  # last lines for context
        raise RuntimeError(f"Sandbox failed (rc={process.returncode}):\n{tail}")

    # parse only the final JSON line
    metrics = json.loads(
        "{" + "".join(full_output[-50:]).split("{")[-1].strip() if full_output else ""
    )

    if "error" in metrics:
        raise RuntimeError(f"Sandbox error: {metrics['error']}")

    return validate_metrics(metrics)


# Insert verified metrics into DB
def log_to_db(
    write_api, hotkey, uid, metrics, config, gist_url, current_block, benchmark_flag
):
    influxdb_measurement = (
        "hotkey_scores" if benchmark_flag == False else "benchmark_scores"
    )
    point = (
        Point(influxdb_measurement)
        .tag("number_of_nodes", config.number_of_nodes)
        .tag("max_steps", config.max_steps)
        .tag("model_size", config.model_size)
        .tag("dataset", config.dataset)
        .tag("current_block", current_block)
        .tag("hotkey", hotkey)
        .tag("uid", uid)
        .tag("gist_url", gist_url)
        .tag("benchmark_flag", benchmark_flag)
        .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)
    )
    for k, v in metrics.items():
        point.field(k, v)
    write_api.write(
        bucket=config.influxdb.bucket, org=config.influxdb.org, record=point
    )


# Validate a single miner: fetch gist, verify hash, run sandbox, log results
def validate_miner(
    hotkey,
    gist_url,
    expected_hash,
    config,
    benchmark=False,
    benchmark_file=None,
    sandbox_logger=None,
):
    """Fetch gist, verify hash, run sandbox, log results"""
    current_strategy_path = "/root/DisTrOpZ/evaluator/sandbox/strategy.py"

    if benchmark == False:
        bt.logging.info(f"Evaluating miner {hotkey}")
        # Fetch gist
        match = re.search(r"([0-9a-fA-F]{8,})$", gist_url)
        # if hotkey not in WHITELISTED_HOTKEYS:
        #     raise ValueError(f"Hotkey {hotkey} not in whitelist")
        if not match:
            raise ValueError(f"Could not parse gist ID from URL: {gist_url}")
        gist_id = match.group(1)
        # breakpoint()
        api_url = f"https://api.github.com/gists/{gist_id}"

        # Use GitHub token if available to avoid rate limiting (60/hr unauthenticated vs 5000/hr authenticated)
        headers = {}
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        resp = requests.get(api_url, headers=headers)

        # Handle rate limiting with retry
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_time = resp.headers.get("X-RateLimit-Reset")
            if reset_time:
                wait_seconds = int(reset_time) - int(time.time()) + 5
                bt.logging.warning(
                    f"Rate limited. Waiting {wait_seconds}s until reset..."
                )
                time.sleep(max(wait_seconds, 60))
                resp = requests.get(api_url, headers=headers)
            else:
                bt.logging.warning("Rate limited. Waiting 60s before retry...")
                time.sleep(60)
                resp = requests.get(api_url, headers=headers)

        last_updated_at = resp.json().get("updated_at")
        time.sleep(15)

        if resp.status_code == 404:
            raise ValueError(f"Gist not found or not public: {gist_url}")
        resp.raise_for_status()

        files = resp.json().get("files", {})
        if not files:
            raise ValueError("Gist has no files.")

        _, fileinfo = next(iter(files.items()))
        code = fileinfo["content"]

        # Verify hash
        actual_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch! expected {expected_hash[:8]}, got {actual_hash[:8]}"
            )

        # Write to temp file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

    else:
        bt.logging.info(f"Benchmarking {benchmark_file}")
        tmp_path = f"/root/DisTrOpZ/miner/{benchmark_file}"
        last_updated_at = datetime.datetime.now().isoformat()

    shutil.copy(tmp_path, current_strategy_path)
    os.chmod(current_strategy_path, 0o644)

    try:
        bt.logging.success(f"✅ Run {current_strategy_path} in sandbox")
        # return {} 17 - 256
        metrics = run_in_sandbox(
            current_strategy_path, config, sandbox_logger=sandbox_logger
        )
        metrics["last_update"] = last_updated_at

        # Cleanup after sandbox execution
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bt.logging.success(f"✅ Miner {hotkey}: {metrics}")
        return metrics

    except Exception as e:
        bt.logging.error(f"❌ Failed miner {hotkey}: {e}")
        traceback.print_exc()
        return {}

    finally:
        if benchmark == False:
            os.remove(tmp_path)


# Main validation loop
def main():
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument(
        "--number_of_nodes",
        type=int,
        default=4,
        help="Nuber of GPUs to use for testing",
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum number of steps to test"
    )
    parser.add_argument("--model_size", default="large", help="NanoGPT model size")
    parser.add_argument("--dataset", default="owt", help="Dataset name")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument("--netuid", type=int, default=38, help="Bittensor network UID.")
    parser.add_argument("--influxdb.measurement", default="mechanism-1")
    parser.add_argument("--influxdb.bucket", required=True)
    parser.add_argument("--influxdb.org", required=True)
    parser.add_argument("--influxdb.url", default="http://localhost:8087")
    parser.add_argument(
        "--influxdb.token", help="InfluxDB token (or set INFLUXDB_TOKEN)"
    )

    # Add wallet arguments
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.Config(parser)
    bt.logging.setLevel("INFO")

    # Set up Loki logging for evaluator (returns logger + listener)
    eval_logger, loki_listener = setup_loki_logging(
        config=config, component="evaluator"
    )

    # Set up separate Loki handler for sandbox logs
    sandbox_logger, sandbox_listener = add_sandbox_handler(config=config)

    # Add subtensor and metagraph arguments
    subtensor = bt.Subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    current_block = subtensor.block
    bt.logging.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")
    eval_logger.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")

    # Set up InfluxDB client
    influx = InfluxDBClient(
        url=config.influxdb.url, token=config.influxdb.token, org=config.influxdb.org
    )
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    read_api = influx.query_api()

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(
        config.output_dir,
        f"metrics-gpt-{config.model_size}-{config.dataset}-{config.number_of_nodes}-{config.max_steps}-{datetime_stamp}.json",
    )

    metrics = {}
    urls = {}

    for benchmark in os.listdir("/root/DisTrOpZ/miner/"):
        if (
            benchmark == "miner_base.py"
            or benchmark == "miner.py"
            or "sparta" in benchmark
        ):
            continue

        bt.logging.info(f"🔍 Running benchmark for {benchmark}")

        try:
            gist_url = "benchmark"
            sha = "benchmark"
            hotkey = f"benchmark_{benchmark.split('miner_')[-1].replace('.py','')}"

            # verify and save
            hotkey_metrics = validate_miner(
                "",
                "",
                "",
                config,
                benchmark=True,
                benchmark_file=benchmark,
                sandbox_logger=sandbox_logger,
            )

            urls[hotkey] = hotkey
            metrics[hotkey] = hotkey_metrics
            bt.logging.success(f"✅ Finished miner {hotkey}: {hotkey_metrics}")

            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            bt.logging.error(f"⚠️ Error validating {hotkey}: {e}")

    while True:
        for uid, hotkey in enumerate(metagraph.hotkeys):
            bt.logging.info(f"🔍 Checking miner UID={uid} hotkey={hotkey}")
            eval_logger.info(f"Checking miner UID={uid} hotkey={hotkey}")
            if int(uid) == 25:
                continue

            try:
                meta = subtensor.get_commitment(netuid=config.netuid, uid=uid)
                if not meta:
                    bt.logging.warning(f"❌ No strategy_gist for {hotkey}")
                    eval_logger.warning(f"No strategy_gist for {hotkey}")
                    continue

                gist_url = meta[64:]
                sha = meta[:64]
                bt.logging.info(f"Found gist: {gist_url}")
                eval_logger.info(f"Found gist: {gist_url}")

                # verify and save
                hotkey_metrics = validate_miner(
                    hotkey, gist_url, sha, config, sandbox_logger=sandbox_logger
                )

                urls[hotkey] = gist_url
                metrics[hotkey] = hotkey_metrics
                bt.logging.success(f"✅ Finished miner {hotkey}: {hotkey_metrics}")
                eval_logger.info(f"Finished miner {hotkey}: {hotkey_metrics}")

                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=2)

            except Exception as e:
                bt.logging.error(f"⚠️ Error validating {hotkey}: {e}")
                eval_logger.error(f"Error validating {hotkey}: {e}")

        # Opening JSON file
        # with open('/root/DisTrOpZ/results/metrics-gpt-medium-owt-4-100-2025-12-25.json') as json_file: metrics = json.load(json_file)
        # urls = {k:k for k in metrics.keys()}
        eval_logger.info("Creating DF")
        df = pd.DataFrame(metrics).T
        eval_logger.info("Calculating scores")
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
        eval_logger.info("Logging to DB")
        benchmark = None
        for hotkey in metrics.keys():
            if hotkey in urls:
                gist_url = urls[hotkey]
                if "benchmark" in hotkey:
                    uid = hotkey
                    benchmark = True
                else:
                    uid = metagraph.hotkeys.index(hotkey)
                    benchmark = False
                # breakpoint()
                metrics[hotkey]["score"] = df.at[hotkey, "score"]
                # breakpoint()
                log_to_db(
                    write_api,
                    hotkey,
                    uid,
                    metrics[hotkey],
                    config,
                    gist_url,
                    current_block,
                    benchmark,
                )
                bt.logging.info(f"Logged {hotkey} metrics")
                eval_logger.info(f"Logged {hotkey} metrics to DB")

        # save results
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        bt.logging.success(f"Saved metrics → {out_path} → {current_block}")
        eval_logger.info(f"Saved metrics to {out_path} at block {current_block}")
        # breakpoint()


if __name__ == "__main__":
    main()
