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
def run_in_sandbox(gist_path, config, sandbox_logger=None, logger=None):
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
    if logger:
        logger.info(f"  🐳 Running sandbox...")
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
            # print(line, end="")  # stream to your terminal in real-time
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


# Fetch the most recent metrics for a hotkey from InfluxDB
def get_cached_metrics(read_api, hotkey, config, logger=None):
    """
    Query InfluxDB for the most recent metrics for a given hotkey.
    Returns the metrics dict if found, or None if no cached data exists.
    Includes the 'gist_sha' field for cache invalidation checks.
    """
    query = f"""
    from(bucket: "{config.influxdb.bucket}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "hotkey_scores")
      |> filter(fn: (r) => r["hotkey"] == "{hotkey}")
      |> filter(fn: (r) => r["number_of_nodes"] == "{config.number_of_nodes}")
      |> filter(fn: (r) => r["max_steps"] == "{config.max_steps}")
      |> filter(fn: (r) => r["model_size"] == "{config.model_size}")
      |> filter(fn: (r) => r["dataset"] == "{config.dataset}")
      |> filter(fn: (r) => exists r["gist_sha"])
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    """

    try:
        tables = read_api.query(query, org=config.influxdb.org)
        for table in tables:
            for record in table.records:
                # gist_sha is stored as a TAG (not field) for reliable string retrieval
                gist_sha = record.values.get("gist_sha")

                # Extract metrics from the record
                cached = {
                    "throughput": int(record.values.get("throughput", 0)),
                    "loss": float(record.values.get("loss", 0.0)),
                    "communication": int(record.values.get("communication", 0)),
                    "last_update": record.values.get("last_update"),
                    "gist_sha": gist_sha if gist_sha else None,
                }
                # Include score if available
                if "score" in record.values:
                    cached["score"] = float(record.values.get("score", 0.0))
                return cached
    except Exception as e:
        if logger:
            logger.warning(f"Failed to query cached metrics: {e}")

    return None


# Insert verified metrics into DB
def log_to_db(
    write_api, hotkey, uid, metrics, config, gist_url, current_block, benchmark_flag
):
    influxdb_measurement = (
        "hotkey_scores" if benchmark_flag == False else "benchmark_scores"
    )
    # Extract gist_sha to store as a tag (string fields don't pivot well in Flux)
    gist_sha = metrics.get("gist_sha")
    # Remove from metrics dict so it's not written as a field (it's stored as tag)
    metrics_to_write = {k: v for k, v in metrics.items() if k != "gist_sha"}

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
        .tag("gist_sha", gist_sha or "")  # Store SHA as tag for reliable retrieval
        .tag("benchmark_flag", benchmark_flag)
        .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)
    )
    for k, v in metrics_to_write.items():
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
    logger=None,
):
    """Fetch gist, verify hash, run sandbox, log results"""
    current_strategy_path = "/root/DisTrOpZ/evaluator/sandbox/strategy.py"

    if benchmark == False:
        # Fetch gist
        match = re.search(r"([0-9a-fA-F]{8,})$", gist_url)
        if not match:
            raise ValueError(f"Could not parse gist ID from URL: {gist_url}")
        gist_id = match.group(1)
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
                if logger:
                    logger.warning(f"  ⏳ Rate limited. Waiting {wait_seconds}s...")
                time.sleep(max(wait_seconds, 60))
                resp = requests.get(api_url, headers=headers)
            else:
                if logger:
                    logger.warning("  ⏳ Rate limited. Waiting 60s...")
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
        if logger:
            logger.info(f"  🔧 Benchmarking {benchmark_file}")
        tmp_path = f"/root/DisTrOpZ/miner/{benchmark_file}"
        last_updated_at = datetime.datetime.now().isoformat()

    shutil.copy(tmp_path, current_strategy_path)
    os.chmod(current_strategy_path, 0o644)

    try:
        metrics = run_in_sandbox(
            current_strategy_path, config, sandbox_logger=sandbox_logger, logger=logger
        )
        metrics["last_update"] = last_updated_at
        metrics["gist_sha"] = expected_hash  # Store SHA for cache invalidation

        # Cleanup after sandbox execution
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        if logger:
            logger.error(f"  ❌ Sandbox failed: {e}")
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
    config = bt.Config(parser)

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
    eval_logger.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")

    # Set up InfluxDB client
    influx = InfluxDBClient(
        url=config.influxdb.url, token=config.influxdb.token, org=config.influxdb.org
    )
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    read_api = influx.query_api()

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    run_benchmark = False
    while True:
        current_block = subtensor.block
        datetime_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(
            config.output_dir,
            f"metrics-gpt-{config.model_size}-{config.dataset}-{config.number_of_nodes}-{config.max_steps}-{datetime_stamp}.json",
        )

        metrics = {}
        urls = {}

        if run_benchmark is True:
            for benchmark in os.listdir("/root/DisTrOpZ/miner/"):
                if (
                    benchmark == "miner_base.py"
                    or benchmark == "miner.py"
                    or "sparta" in benchmark
                ):
                    continue

                eval_logger.info(f"🔍 Running benchmark for {benchmark}")

                try:
                    gist_url = "benchmark"
                    sha = "benchmark"
                    hotkey = (
                        f"benchmark_{benchmark.split('miner_')[-1].replace('.py','')}"
                    )

                    # verify and save
                    hotkey_metrics = validate_miner(
                        "",
                        "",
                        "",
                        config,
                        benchmark=True,
                        benchmark_file=benchmark,
                        sandbox_logger=sandbox_logger,
                        logger=eval_logger,
                    )

                    urls[hotkey] = hotkey
                    metrics[hotkey] = hotkey_metrics
                    eval_logger.info(f"✅ Finished benchmark {hotkey}")

                    with open(out_path, "w") as f:
                        json.dump(metrics, f, indent=2)

                except Exception as e:
                    eval_logger.error(f"⚠️ Error validating {hotkey}: {e}")

        for uid, hotkey in enumerate(metagraph.hotkeys):
            eval_logger.info(f"")
            eval_logger.info(f"{'═' * 70}")
            eval_logger.info(f"  UID {uid} │ {hotkey[:20]}...{hotkey[-8:]}")
            eval_logger.info(f"{'═' * 70}")

            if int(uid) == 25:
                eval_logger.info(f"  ⏭️  Skipping UID 25")
                continue

            try:
                meta = subtensor.get_commitment(netuid=config.netuid, uid=uid)
                if not meta:
                    eval_logger.warning(f"  ❌ No strategy_gist found")
                    continue

                gist_url = meta[64:]
                sha = meta[:64]
                eval_logger.info(f"  📎 Gist: {gist_url}")
                eval_logger.info(f"  🔑 SHA:  {sha[:16]}...")

                # Check if we have cached metrics with matching SHA (no GitHub API call needed!)
                cached_metrics = get_cached_metrics(
                    read_api, hotkey, config, logger=eval_logger
                )

                if cached_metrics:
                    cached_sha = cached_metrics.get("gist_sha")
                    if cached_sha and cached_sha == sha:
                        eval_logger.info(f"  ♻️  CACHED - SHA unchanged ({sha[:8]}...)")
                        eval_logger.info(
                            f"  └─ throughput={cached_metrics.get('throughput'):,}, loss={cached_metrics.get('loss'):.4f}"
                        )
                        eval_logger.info(f"{'─' * 70}")
                        urls[hotkey] = gist_url
                        metrics[hotkey] = cached_metrics
                        continue
                    elif cached_sha:
                        eval_logger.info(
                            f"  🔄 SHA CHANGED: {cached_sha[:8]}... → {sha[:8]}..."
                        )
                    else:
                        eval_logger.info(f"  📦 Legacy cache (no SHA) - re-evaluating")
                else:
                    eval_logger.info(f"  🆕 No cache found - first evaluation")

                # verify and save
                hotkey_metrics = validate_miner(
                    hotkey,
                    gist_url,
                    sha,
                    config,
                    sandbox_logger=sandbox_logger,
                    logger=eval_logger,
                )

                urls[hotkey] = gist_url
                metrics[hotkey] = hotkey_metrics

                eval_logger.info(f"  ┌─ RESULTS ─────────────────────────────────────")
                eval_logger.info(
                    f"  │ Throughput:    {hotkey_metrics.get('throughput', 0):>12,} tokens/sec"
                )
                eval_logger.info(
                    f"  │ Loss:          {hotkey_metrics.get('loss', 0):>12.4f}"
                )
                eval_logger.info(
                    f"  │ Communication: {hotkey_metrics.get('communication', 0):>12,} bytes"
                )
                eval_logger.info(f"  └───────────────────────────────────────────────")
                eval_logger.info(f"  ✅ EVALUATION COMPLETE")
                eval_logger.info(f"{'─' * 70}")

                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=2)

            except Exception as e:
                eval_logger.error(f"  ❌ ERROR: {e}")
                eval_logger.info(f"{'─' * 70}")

        # Opening JSON file
        # with open('/root/DisTrOpZ/results/metrics-gpt-medium-owt-4-100-2025-12-25.json') as json_file: metrics = json.load(json_file)
        # urls = {k:k for k in metrics.keys()}
        eval_logger.info("Creating DF")
        df = pd.DataFrame(metrics).T
        eval_logger.info("Calculating scores")
        metric_names = ["throughput", "loss", "communication"]

        # min-max normalize each metric
        range_vals = df[metric_names].max() - df[metric_names].min()
        # Handle single data point case (avoid division by zero)
        range_vals = range_vals.replace(0, 1)
        norm = (df[metric_names] - df[metric_names].min()) / range_vals

        df["throughput_norm"] = norm["throughput"].fillna(0.5)
        df["loss_norm"] = (1 - norm["loss"]).fillna(0.5)  # lower loss is better
        df["communication_norm"] = (1 - norm["communication"]).fillna(
            0.5
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
                eval_logger.info(f"Logged {hotkey[:16]}... to DB")

        # save results
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        eval_logger.info(f"✅ Saved metrics → {out_path} → block {current_block}")


if __name__ == "__main__":
    main()
