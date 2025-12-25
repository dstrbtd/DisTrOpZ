import argparse, os, sys, bittensor as bt
from influxdb_client import InfluxDBClient
import pandas as pd
import json
import time
import datetime
import logging

# Add parent directory to path to import logger
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluator.logger import setup_loki_logging

STARTING_LOSS = 11.00


def fetch_metrics_for_all_hotkeys(client, config):
    # Step 1: Find the maximum block number for these config parameters (no restrictive time range needed)
    influxdb_measurement = "hotkeys_scores"
    max_block_query = f"""
    from(bucket:"{config.influxdb.bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r["_measurement"] == "{influxdb_measurement}")
      |> filter(fn: (r) => r["number_of_nodes"] == "{config.number_of_nodes}")
      |> filter(fn: (r) => r["max_steps"] == "{config.max_steps}")
      |> filter(fn: (r) => r["model_size"] == "{config.model_size}")
      |> filter(fn: (r) => r["dataset"] == "{config.dataset}")
      |> filter(fn: (r) => r["benchmark_flag"] == "False")
      |> distinct(column: "current_block")
      |> sort(columns: ["current_block"], desc: true)
      |> limit(n: 1)
    """

    # Execute query to get max block
    max_block_tables = client.query_api().query(
        org=config.influxdb.org, query=max_block_query
    )
    max_block = None
    for table in max_block_tables:
        for record in table.records:
            max_block = record.values.get("current_block")
            break
        if max_block:
            break

    if max_block is None:
        return pd.DataFrame()

    # Step 2: Fetch only data for that maximum block and config parameters
    query = f"""
    from(bucket:"{config.influxdb.bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r["_measurement"] == "{influxdb_measurement}")
      |> filter(fn: (r) => r["number_of_nodes"] == "{config.number_of_nodes}")
      |> filter(fn: (r) => r["max_steps"] == "{config.max_steps}")
      |> filter(fn: (r) => r["model_size"] == "{config.model_size}")
      |> filter(fn: (r) => r["dataset"] == "{config.dataset}")
      |> filter(fn: (r) => r["benchmark_flag"] == "False")
      |> filter(fn: (r) => r["current_block"] == "{max_block}")
    """

    tables = client.query_api().query(org=config.influxdb.org, query=query)
    data = []
    for table in tables:
        for record in table.records:
            data.append(
                {
                    "time": record.get_time(),
                    "block": record.values.get("current_block"),
                    "hotkey": record.values.get("hotkey"),
                    "field": record.get_field(),
                    "value": record.get_value(),
                }
            )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    print(df.pivot(index="hotkey", columns="field", values="value").reset_index())
    return df.pivot(index="hotkey", columns="field", values="value").reset_index()


def fetch_benchmark_scores(client, config):
    # Step 1: Find the maximum block number for these config parameters (no restrictive time range needed)
    influxdb_measurement = "benchmark_scores"
    max_block_query = f"""
    from(bucket:"{config.influxdb.bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r["_measurement"] == "{influxdb_measurement}")
      |> filter(fn: (r) => r["number_of_nodes"] == "{config.number_of_nodes}")
      |> filter(fn: (r) => r["max_steps"] == "{config.max_steps}")
      |> filter(fn: (r) => r["model_size"] == "{config.model_size}")
      |> filter(fn: (r) => r["dataset"] == "{config.dataset}")
      |> filter(fn: (r) => r["benchmark_flag"] == "True")
      |> distinct(column: "current_block")
      |> sort(columns: ["current_block"], desc: true)
      |> limit(n: 1)
    """

    # Execute query to get max block
    max_block_tables = client.query_api().query(
        org=config.influxdb.org, query=max_block_query
    )
    max_block = None
    for table in max_block_tables:
        for record in table.records:
            max_block = record.values.get("current_block")
            break
        if max_block:
            break

    if max_block is None:
        return pd.DataFrame()

    # Step 2: Fetch benchmarks for that maximum block and config parameters
    query = f"""
    from(bucket:"{config.influxdb.bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r["_measurement"] == "{influxdb_measurement}")
      |> filter(fn: (r) => r["number_of_nodes"] == "{config.number_of_nodes}")
      |> filter(fn: (r) => r["max_steps"] == "{config.max_steps}")
      |> filter(fn: (r) => r["model_size"] == "{config.model_size}")
      |> filter(fn: (r) => r["dataset"] == "{config.dataset}")
      |> filter(fn: (r) => r["benchmark_flag"] == "True")
      |> filter(fn: (r) => r["current_block"] == "{max_block}")
    """

    tables = client.query_api().query(org=config.influxdb.org, query=query)
    data = []
    for table in tables:
        for record in table.records:
            data.append(
                {
                    "time": record.get_time(),
                    "block": record.values.get("current_block"),
                    "hotkey": record.values.get("hotkey"),
                    "field": record.get_field(),
                    "value": record.get_value(),
                }
            )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df.pivot(index="hotkey", columns="field", values="value").reset_index()
    print(df)

    benchmark_loss_score = min(df["loss"])
    benchmark_communication_score = min(df["communication"])
    benchmark_throughput_score = max(df["throughput"])

    return (
        benchmark_loss_score,
        benchmark_communication_score,
        benchmark_throughput_score,
    )


def main():
    parser = argparse.ArgumentParser(description="Simple validator script")
    parser.add_argument("--netuid", type=int, default=220)
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
        "--winning_delta",
        type=float,
        default=0.1,
        help="The percentage a new winner has to exceed the previous winner buy",
    )
    parser.add_argument("--influxdb.bucket", required=True)
    parser.add_argument("--influxdb.org", required=True)
    parser.add_argument("--influxdb.url", default="http://localhost:8086")
    parser.add_argument(
        "--influxdb.token", help="InfluxDB token (or set INFLUXDB_TOKEN)"
    )

    # Add wallet arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)
    bt.logging.setLevel("INFO")

    # Set up Loki logging for validator
    loki_listener = setup_loki_logging(config=config, component="validator")

    # Add subtensor and metagraph arguments
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Loaded metagraph with {len(metagraph.hotkeys)} miners.")

    # Load state
    state_path = (
        os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                config.netuid,
                "validator",
            )
        )
        + "/state.json"
    )
    os.makedirs(state_path.split("/state.json")[0], exist_ok=True)

    # Setup influxdb
    token = config.influxdb.token or os.getenv("INFLUXDB_TOKEN")
    if not token:
        sys.exit("ERROR: Provide --token or set INFLUXDB_TOKEN")
    client = InfluxDBClient(
        url=config.influxdb.url, token=config.influxdb.token, org=config.influxdb.org
    )

    while True:
        df = fetch_metrics_for_all_hotkeys(
            client,
            config,
        )
        if df.empty:
            bt.logging.info("No metrics found in InfluxDB. Waiting for 60 seconds.")
            time.sleep(60)

        metrics = ["throughput", "communication", "loss", "last_update"]
        df = df.dropna(subset=metrics, how="any")
        if df.empty:
            sys.exit("Not enough valid metrics to evaluate.")

        # Convert last_update_dt to datetime
        df["last_update_dt"] = pd.to_datetime(df["last_update"], utc=True)

        current_winner_hotkey = (
            df.sort_values(
                by=["communication", "last_update_dt"],
                ascending=[True, True],  # score ↓, last_update ↑
            )
            .iloc[0]
            .hotkey
        )
        current_winner_score = df.loc[
            df.hotkey == current_winner_hotkey, "score"
        ].item()
        current_winner_loss_score = df.loc[
            df.hotkey == current_winner_hotkey, "loss"
        ].item()
        current_winner_communication_score = df.loc[
            df.hotkey == current_winner_hotkey, "communication"
        ].item()
        current_winner_throughput_score = df.loc[
            df.hotkey == current_winner_hotkey, "throughput"
        ].item()

        (
            benchmark_loss_score,
            benchmark_communication_score,
            benchmark_throughput_score,
        ) = fetch_benchmark_scores(client, config)

        if (
            (current_winner_loss_score <= benchmark_loss_score)
            or (current_winner_communication_score <= benchmark_communication_score)
            or (current_winner_throughput_score >= benchmark_throughput_score)
        ):
            # Burn hotkey
            current_winner_hotkey = "5ECDEtiHDP7tXeG3L7PViHsjSUPCsijKEokrFWhdXuATDjH1"
            current_winner_score = 0.5

            # Save current winner data
            current_winner_data = {
                "throughput": 67409,
                "loss": 10.789522171020508,
                "communication": 353439776,
                "score": current_winner_score,
                "last_update": datetime.datetime.now().isoformat(),
            }

        else:
            # If state is saved compare to previous winner
            if os.path.isfile(state_path):
                with open(state_path, "r") as file:
                    state = json.load(file)

                previous_winner_hotkey = state[0]["hotkey"]
                if current_winner_hotkey != previous_winner_hotkey:
                    previous_winner_score = df.loc[
                        df.hotkey == previous_winner_hotkey, "score"
                    ].item()
                    if (
                        (current_winner_score - previous_winner_score)
                        / previous_winner_score
                    ) <= config.winning_delta:
                        bt.logging.info(
                            f"Current winer: {current_winner_hotkey} score={current_winner_score:.3f} not greater than previous winner: {previous_winner_hotkey} score: {previous_winner_hotkey} by 10%"
                        )
                        current_winner_hotkey = previous_winner_hotkey
                        current_winner_score = previous_winner_score

            # Save current winner data
            current_winner_data = df.loc[df.hotkey == current_winner_hotkey, :].to_dict(
                orient="records"
            )

            for record in current_winner_data:
                ts = record.get("last_update_dt")
                if isinstance(ts, pd.Timestamp):
                    record["last_update_dt"] = ts.isoformat()

        with open(state_path, "w") as json_file:
            json.dump(current_winner_data, json_file, indent=4)

        bt.logging.info(
            f"🏆 Top miner: {current_winner_hotkey} (score={current_winner_score:.3f})"
        )

        # Set weights
        weights = [0.0] * len(metagraph.hotkeys)
        if current_winner_hotkey in metagraph.hotkeys:
            idx = metagraph.hotkeys.index(current_winner_hotkey)
            weights[idx] = 1.0
            bt.logging.info(f"Setting weight=1 for {current_winner_hotkey}")
            subtensor.set_weights(
                wallet=wallet,
                netuid=config.netuid,
                weights=weights,
                uids=list(range(len(weights))),
            )
        else:
            bt.logging.warning(
                f"Hotkey {current_winner_hotkey} not found in metagraph."
            )

        metagraph.sync(subtensor=subtensor)
        time.sleep(120)


if __name__ == "__main__":
    main()
