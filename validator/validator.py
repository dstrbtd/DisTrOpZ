import argparse, os, sys, bittensor as bt
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np


def fetch_metrics(client, bucket, org, measurement):
    query = f'from(bucket:"{bucket}") |> range(start: -24h) |> filter(fn: (r) => r["_measurement"] == "{measurement}")'
    tables = client.query_api().query(org=org, query=query)
    data = []
    for table in tables:
        for record in table.records:
            data.append(
                {
                    "time": record.get_time(),
                    "hotkey": record.values.get("hotkey"),
                    "field": record.get_field(),
                    "value": record.get_value(),
                }
            )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df.pivot_table(
        index="hotkey", columns="field", values="value", aggfunc="mean"
    ).reset_index()


def normalize(df, columns, invert_loss=True):
    df = df.copy()
    for c in columns:
        if invert_loss and c == "loss":
            df[c] = -df[c]  # lower loss → higher score
        col_min, col_max = df[c].min(), df[c].max()
        if col_max != col_min:
            df[c] = (df[c] - col_min) / (col_max - col_min)
        else:
            df[c] = 0.5  # neutral if all values equal
    df["score"] = df[columns].mean(axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Simple validator script")
    parser.add_argument("--netuid", type=int, default=220)
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

    bt.logging.setLevel("INFO")

    token = config.influxdb.token or os.getenv("INFLUXDB_TOKEN")
    if not token:
        sys.exit("ERROR: Provide --token or set INFLUXDB_TOKEN")

    client = InfluxDBClient(
        url=config.influxdb.url, token=config.influxdb.token, org=config.influxdb.org
    )
    df = fetch_metrics(
        client, config.influxdb.bucket, config.influxdb.org, config.influxdb.measurement
    )

    if df.empty:
        sys.exit("No metrics found in InfluxDB.")

    metrics = ["throughput", "communication", "loss"]
    df = df.dropna(subset=metrics, how="any")
    if df.empty:
        sys.exit("Not enough valid metrics to evaluate.")

    df = normalize(df, metrics)
    best = df.loc[df["score"].idxmax()]
    best_hotkey = best["hotkey"]
    bt.logging.info(f"🏆 Top miner: {best_hotkey} (score={best['score']:.3f})")

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    weights = [0.0] * len(metagraph.hotkeys)
    if best_hotkey in metagraph.hotkeys:
        idx = metagraph.hotkeys.index(best_hotkey)
        weights[idx] = 1.0
        bt.logging.info(f"Setting weight=1 for {best_hotkey}")
        subtensor.set_weights(
            wallet=wallet,
            netuid=config.netuid,
            weights=weights,
            uids=list(range(len(weights))),
        )
    else:
        bt.logging.warning(f"Hotkey {best_hotkey} not found in metagraph.")


if __name__ == "__main__":
    main()
