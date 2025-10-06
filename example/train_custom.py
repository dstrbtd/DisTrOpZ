import torch
from nanogpt import GPT, GPTConfig, get_dataset

from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.demo import DeMoStrategy
from exogym.strategy.optim import OptimSpec
from exogym.strategy.miner_diloco import STRATEGY as diloco_strategy
from exogym.strategy.miner_demo import STRATEGY as demo_strategy
from exogym.trainer import Trainer

NUM_NODES = 2
DEVICE = "mps"


def main():
    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        "shakespeare",
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        "shakespeare", block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )

    # Create model
    gpt_config = GPTConfig.gpt2_size_map("small")
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        # port=12355 # Modify this if we get port conflict errors
    )

    # diloco_strategy = DiLoCoStrategy(
    #     optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
    #     lr_scheduler="lambda_cosine",
    #     lr_scheduler_kwargs={
    #         "warmup_steps": 1000,
    #         "cosine_anneal": True,
    #     },
    #     max_norm=1.0,
    #     H=5,
    # )

    # demo_strategy = DeMoStrategy(
    #     lr_scheduler="lambda_cosine",
    #     lr_scheduler_kwargs={
    #         "warmup_steps": 1000,
    #         "cosine_anneal": True,
    #     },
    # )

    strategies = [diloco_strategy, demo_strategy]
    
    for strategy in strategies:
        # Train
        print(
            trainer.fit(
                max_steps=20,
                strategy=strategy,
                num_epochs=5,
                num_nodes=NUM_NODES,
                device=DEVICE,
                batch_size=16,
                minibatch_size=8,  # Gradient accumulation to ensure we can fit in memory for a 96GB machine. Make this even lower for smaller devices.
                shuffle=False,
                val_size=256,
                val_interval=10,
                # wandb_project='50M-GPT',
                run_name="diloco-H100",
            )
        )


if __name__ == "__main__":
    main()
