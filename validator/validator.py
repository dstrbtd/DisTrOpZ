import argparse, os, sys, bittensor as bt
from influxdb_client import InfluxDBClient
import pandas as pd
import json
import time


def fetch_metrics(client, bucket, org, measurement):
    query = f'from(bucket:"{bucket}") |> range(start: -24h) |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
    tables = client.query_api().query(org=org, query=query)
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
    df = df[df.block == df.block.max()]
    return df.pivot(index="hotkey", columns="field", values="value").reset_index()


def main():
    parser = argparse.ArgumentParser(description="Simple validator script")
    parser.add_argument("--netuid", type=int, default=220)
    parser.add_argument(
        "--winning_delta",
        type=float,
        default=0.1,
        help="The percentage a new winner has to exceed the previous winner buy",
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
    bt.logging.add_args(parser)
    config = bt.config(parser)
    bt.logging.setLevel("INFO")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

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
        df = fetch_metrics(
            client,
            config.influxdb.bucket,
            config.influxdb.org,
            config.influxdb.measurement,
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
                by=["score", "last_update_dt"],
                ascending=[False, True],  # score ↓, last_update ↑
            )
            .iloc[0]
            .hotkey
        )
        current_winner_score = df.loc[
            df.hotkey == current_winner_hotkey, "score"
        ].item()

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
