import logging.handlers

# Patch stdlib's QueueListener to ignore EOFError
if not hasattr(logging.handlers.QueueListener, "_orig_monitor"):
    orig = logging.handlers.QueueListener._monitor

    def _safe_monitor(self):
        try:
            orig(self)
        except (EOFError, OSError, BrokenPipeError):
            return

    logging.handlers.QueueListener._monitor = _safe_monitor

import json, os, importlib.util, sys, traceback
from exogym.trainer import Trainer
from nanogpt import GPT, GPTConfig, get_dataset
import bittensor as bt
import sys

NUM_NODES = int(os.getenv("NUM_NODES", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2"))
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
DATASET = os.getenv("DATASET", "shakespeare")
DEVICE = os.getenv("DEVICE", "cuda")

print("MAX_STEPS", MAX_STEPS)
print("NUM_NODES", NUM_NODES)


def load_strategy(path):
    spec = importlib.util.spec_from_file_location("miner_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "STRATEGY"):
        raise ValueError("No STRATEGY found in submitted code.")
    return module.STRATEGY


def main():
    # Miner script is mounted at /sandbox/strategy.py
    script_path = "sandbox/strategy.py"
    script_path = "/root/DisTrOpZ/evaluator/sandbox/strategy.py"
    # script_path = "/root/DisTrOpZ/miner/miner_sparseloco.py"
    try:
        strategy = script_path
        train_dataset, _ = get_dataset(
            DATASET,
            block_size=1024,
            device="cpu",
            start_pc=0.0,
            end_pc=0.005 * NUM_NODES,
        )
        val_dataset, _ = get_dataset(
            DATASET, block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
        )
        model = GPT(GPTConfig.gpt2_size_map(MODEL_SIZE))
        trainer = Trainer(model, train_dataset, val_dataset, device=DEVICE)

        _, metrics_out = trainer.fit(
            max_steps=MAX_STEPS,
            strategy=strategy,
            num_epochs=1,
            num_nodes=NUM_NODES,
            device=DEVICE,
            batch_size=256,
            minibatch_size=32,
            shuffle=False,
            val_size=256,
            val_interval=10,
        )

        # Extract the 3 integer metrics
        result = {
            "throughput": int(metrics_out.get("tokens_per_sec", 0)),
            "loss": float(metrics_out.get("loss_per_token", 0.0)),
            "communication": int(metrics_out.get("comm_bytes_total", 0)),
        }
        print("\n" + json.dumps(result))  # output to stdout

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
