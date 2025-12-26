# evaluator/precache_dataset.py
from nanogpt import get_dataset
import os

DATASET = os.getenv("DATASET", "owt")
NUM_NODES = int(os.getenv("NUM_NODES", 4))

print(f"📦 Pre-caching dataset: {DATASET} for {NUM_NODES} nodes...")

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

print("✅ Dataset cached successfully.")
