import logging.handlers
import torch
import gc
import logging
import os

# Patch stdlib's QueueListener to ignore EOFError
if not hasattr(logging.handlers.QueueListener, "_orig_monitor"):
    orig = logging.handlers.QueueListener._monitor

    def _safe_monitor(self):
        try:
            orig(self)
        except (EOFError, OSError, BrokenPipeError):
            return

    logging.handlers.QueueListener._monitor = _safe_monitor

import json, importlib.util, sys, traceback
from exogym.trainer import Trainer
from nanogpt import GPT, GPTConfig, get_dataset
import bittensor as bt

# Set up logging - will output to stdout which gets captured by parent process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

NUM_NODES = int(os.getenv("NUM_NODES", "2"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2"))
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
DATASET = os.getenv("DATASET", "shakespeare")
DEVICE = os.getenv("DEVICE", "cuda")

logger.info(f"MAX_STEPS: {MAX_STEPS}")
logger.info(f"NUM_NODES: {NUM_NODES}")


def load_strategy(path):
    spec = importlib.util.spec_from_file_location("miner_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "STRATEGY"):
        raise ValueError("No STRATEGY found in submitted code.")
    return module.STRATEGY


def main():
    # Miner script is mounted at /sandbox/strategy.py
    script_path = "/root/DisTrOpZ/evaluator/sandbox/strategy.py"
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
            minibatch_size=2,
            shuffle=False,
            val_size=256,
            val_interval=10,
        )

        # Extract the 3 integer metrics
        result = {
            "throughput": int(metrics_out.get("tokens_per_sec", 0)),
            "loss": float(metrics_out.get("eval_loss", 0.0)),
            "communication": int(metrics_out.get("comm_bytes_total", 0)),
        }
        logger.info(f"Training completed. Metrics: {result}")
        print("\n" + json.dumps(result))  # output to stdout for parent process to parse

    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}", exc_info=True)
        print(
            json.dumps({"error": str(e)})
        )  # output to stdout for parent process to parse
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:
        # Explicit cleanup
        try:
            if "model" in locals():
                del model
            if "trainer" in locals():
                del trainer
            if "train_dataset" in locals():
                del train_dataset
            if "val_dataset" in locals():
                del val_dataset
        except:
            pass
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


if __name__ == "__main__":
    main()
