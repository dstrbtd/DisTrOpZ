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
    return r.json()["id"]


def commit_to_chain(gist_url, digest, netuid, wallet, subtensor):
    return subtensor.set_commitment(wallet, netuid, digest + gist_url)


def main():
    parser = argparse.ArgumentParser(description="Miner script")
    parser.add_argument("--script.path", help="Path to strategy .py")
    parser.add_argument(
        "--script.desc", default="Distributed training strategy submission"
    )
    parser.add_argument(
        "--github.token", help="GitHub token (falls back to $GITHUB_TOKEN)"
    )
    parser.add_argument("--no-commit", action="store_true", help="Skip on-chain commit")
    parser.add_argument(
        "--netuid", type=int, default=220, help="Bittensor network UID."
    )
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    config = bt.Config(parser)

    token = config.github.token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise Exception("ERROR: Provide --github.token or set GITHUB_TOKEN")

    bt.logging.setLevel("INFO")
    bt.logging.info(config)
    wallet = bt.Wallet(config=config)
    subtensor = bt.Subtensor(config=config)

    if not subtensor.is_hotkey_registered(
        netuid=config.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address,
    ):
        bt.logging.error(
            f"Wallet: {wallet} is not registered on netuid {config.netuid}."
            f" Please register the hotkey using `btcli subnets register` before trying again"
        )

    if not os.path.exists(config.script.path):
        raise Exception(f"ERROR: {config.script.path} not found")

    digest = sha256_file(config.script.path)
    gist_url = create_gist(config.script.path, token, config.script.desc)

    bt.logging.info(f"✅ Gist: {gist_url}\n🔒 SHA256: {digest}")

    if not config.no_commit:
        commit_to_chain(gist_url, digest, config.netuid, wallet, subtensor)
        bt.logging.info("🔗 Committed to chain.")


if __name__ == "__main__":
    main()
