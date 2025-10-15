import json
from nanogpt import GPT, GPTConfig, get_dataset

from miner.miner_demo import STRATEGY as demo_strategy
from miner.miner_diloco import STRATEGY as diloco_strategy
from miner.miner_federated_averaging import STRATEGY as federated_averaging_strategy
from miner.miner_sparta_diloco import STRATEGY as sparta_diloco_strategy
from miner.miner_sparta import STRATEGY as sparta_strategy
from exogym.trainer import Trainer

NUM_NODES = 2
DEVICE = "cuda"
MAX_STEPS = 10000
MODEL_SIZE = "large"

def main():
    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        "owt",
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        "owt", block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )

    # Create model
    gpt_config = GPTConfig.gpt2_size_map(MODEL_SIZE)

    strategies = [diloco_strategy, demo_strategy, federated_averaging_strategy, sparta_diloco_strategy, sparta_strategy]
    # strategies = [demo_strategy]

    metrics = {}
    for strategy in strategies:
        # Create model
        model = GPT(gpt_config)

        # Create trainer
        trainer = Trainer(
            model,
            train_dataset,
            val_dataset,
            # port=12355 # Modify this if we get port conflict errors
        )
        
        # Train
        _, metrics_out = trainer.fit(
            max_steps=MAX_STEPS,
            strategy=strategy,
            num_epochs=5,
            num_nodes=NUM_NODES,
            device=DEVICE,
            batch_size=16,
            minibatch_size=2,
            shuffle=False,
            val_size=256,
            val_interval=10,
            run_name=str(strategy),
        )
        metrics[str(strategy)] = metrics_out
        
        with open(f"results/metrics-gpt-{MODEL_SIZE}-{NUM_NODES}-{MAX_STEPS}.json", 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

if __name__ == "__main__":
    main()
