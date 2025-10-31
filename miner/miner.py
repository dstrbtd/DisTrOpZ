import argparse, os, json, requests, bittensor as bt, hashlib, sys


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_gist(file_path, token, desc):
    payload = {
        "description": desc,
        "public": True,
        "files": {os.path.basename(file_path): {"content": open(file_path).read()}},
    }
    r = requests.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
        data=json.dumps(payload),
    )
    r.raise_for_status()
    return r.json()["html_url"]


def commit_to_chain(gist_url, digest, netuid, wallet, subtensor):
    return subtensor.commit(wallet, netuid, digest + gist_url)


def main():
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument("--script_path", help="Path to strategy .py")
    parser.add_argument(
        "--script_desc", default="Distributed training strategy submission"
    )
    parser.add_argument(
        "--github-token", help="GitHub token (falls back to $GITHUB_TOKEN)"
    )
    parser.add_argument("--no-commit", action="store_true", help="Skip on-chain commit")
    parser.add_argument(
        "--netuid", type=int, default=220, help="Bittensor network UID."
    )
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = "test"
    config.subtensor.chain_endpoint = "wss://test.finney.opentensor.ai:443/"

    token = config.github_token or os.getenv("GITHUB_TOKEN")
    if not token:
        sys.exit("ERROR: Provide --github-token or set GITHUB_TOKEN")

    bt.logging.setLevel("INFO")
    bt.logging.info(config)
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)

    if not os.path.exists(config.script_path):
        sys.exit(f"ERROR: {config.script_path} not found")

    digest = sha256_file(config.script_path)
    gist_url = create_gist(config.script_path, token, config.script_desc)
    # gist_url = "https://gist.github.com/KMFODA/547bb877893319071d6e92f4892f8986"
    bt.logging.info(f"✅ Gist: {gist_url}\n🔒 SHA256: {digest}")

    if not config.no_commit:
        commit_to_chain(gist_url, digest, config.netuid, wallet, subtensor)
        bt.logging.info("🔗 Committed to chain.")


if __name__ == "__main__":
    main()
