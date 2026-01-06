#!/usr/bin/env python3
"""
DisTrOpZ Evaluation Sandbox

This script can be run in two modes:
1. SANDBOX MODE (default): Run inside Docker container by evaluator.py
   - Outputs JSON metrics to stdout for parsing
   - Configured via environment variables

2. LOCAL MODE: Run locally to test your strategy against Hurdle Rates
   - Usage: python evaluation_sandbox.py --local --strategy /path/to/your_strategy.py
   - Displays formatted results and comparison to Hurdle Rates
   - Hurdle rates can be customized via CLI arguments

Example Local Usage:
    # Basic test with default hurdle rates
    python evaluation_sandbox.py --local --strategy ./my_strategy.py
    
    # Custom configuration
    python evaluation_sandbox.py --local --strategy ./my_strategy.py --num-nodes 4 --max-steps 1000
    
    # Custom hurdle rates
    python evaluation_sandbox.py --local --strategy ./my_strategy.py \\
        --hurdle-loss 5.0 --hurdle-comm 50000000000 --hurdle-throughput 10000
"""

import logging.handlers
import torch
import gc
import logging
import os
import argparse
import sys

# Patch stdlib's QueueListener to ignore EOFError
if not hasattr(logging.handlers.QueueListener, "_orig_monitor"):
    orig = logging.handlers.QueueListener._monitor

    def _safe_monitor(self):
        try:
            orig(self)
        except (EOFError, OSError, BrokenPipeError):
            return

    logging.handlers.QueueListener._monitor = _safe_monitor

import json, importlib.util, traceback
from exogym.trainer import Trainer
from nanogpt import GPT, GPTConfig, get_dataset

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT HURDLE RATES - Minimum thresholds your strategy should beat
# Users can override these via CLI arguments when running locally
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_HURDLE_RATES = {
    "throughput": 211_876,  # tokens/sec - higher is better
    "loss": 6.35,  # eval loss - lower is better
    "communication": 70_264_572_848,  # bytes - lower is better
}


def load_strategy(path):
    """Load a strategy module from a file path."""
    spec = importlib.util.spec_from_file_location("miner_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "STRATEGY"):
        raise ValueError("No STRATEGY found in submitted code.")
    return module.STRATEGY


def format_bytes(num_bytes):
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_local_results(metrics, hurdle_rates, config):
    """Print formatted results for local testing mode."""

    # ANSI colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    print()
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  DisTrOpZ Strategy Evaluation Results{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")
    print()

    # Configuration summary
    print(f"{DIM}Configuration:{RESET}")
    print(f"  • Nodes:      {config['num_nodes']}")
    print(f"  • Max Steps:  {config['max_steps']}")
    print(f"  • Model Size: {config['model_size']}")
    print(f"  • Dataset:    {config['dataset']}")
    print(f"  • Device:     {config['device']}")
    print()

    print(f"{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}  Metric Results vs Hurdle Rates{RESET}")
    print(f"{BOLD}{'─' * 70}{RESET}")
    print()

    all_passed = True

    # Throughput (higher is better)
    throughput = metrics.get("throughput", 0)
    hurdle_throughput = hurdle_rates["throughput"]
    if hurdle_throughput == 0:
        throughput_passed = True
        throughput_status = f"{DIM}(no hurdle set){RESET}"
        throughput_pct_str = "N/A"
    else:
        throughput_passed = throughput >= hurdle_throughput
        throughput_status = (
            f"{GREEN}✓ PASS{RESET}" if throughput_passed else f"{RED}✗ FAIL{RESET}"
        )
        throughput_pct = throughput / hurdle_throughput * 100
        throughput_pct_str = f"{throughput_pct:.1f}% of hurdle"
    all_passed = all_passed and throughput_passed

    print(f"  {BOLD}Throughput{RESET} (higher is better)")
    print(f"    Your result:  {throughput:,.0f} tokens/sec")
    print(f"    Hurdle rate:  {hurdle_throughput:,.0f} tokens/sec")
    print(f"    Performance:  {throughput_pct_str}  {throughput_status}")
    print()

    # Loss (lower is better)
    loss = metrics.get("loss", float("inf"))
    hurdle_loss = hurdle_rates["loss"]
    if hurdle_loss == 0:
        loss_passed = True
        loss_status = f"{DIM}(no hurdle set){RESET}"
        loss_pct_str = "N/A"
    else:
        loss_passed = loss <= hurdle_loss
        loss_status = f"{GREEN}✓ PASS{RESET}" if loss_passed else f"{RED}✗ FAIL{RESET}"
        loss_pct = (hurdle_loss / loss * 100) if loss > 0 else 0
        loss_pct_str = f"{loss_pct:.1f}% better than hurdle"
    all_passed = all_passed and loss_passed

    print(f"  {BOLD}Loss{RESET} (lower is better)")
    print(f"    Your result:  {loss:.4f}")
    print(f"    Hurdle rate:  {hurdle_loss:.4f}")
    print(f"    Performance:  {loss_pct_str}  {loss_status}")
    print()

    # Communication (lower is better)
    comm = metrics.get("communication", float("inf"))
    hurdle_comm = hurdle_rates["communication"]
    if hurdle_comm == 0:
        comm_passed = True
        comm_status = f"{DIM}(no hurdle set){RESET}"
        comm_pct_str = "N/A"
    else:
        comm_passed = comm <= hurdle_comm
        comm_status = f"{GREEN}✓ PASS{RESET}" if comm_passed else f"{RED}✗ FAIL{RESET}"
        comm_pct = (hurdle_comm / comm * 100) if comm > 0 else 0
        comm_pct_str = f"{comm_pct:.1f}% better than hurdle"
    all_passed = all_passed and comm_passed

    print(f"  {BOLD}Communication{RESET} (lower is better)")
    print(f"    Your result:  {format_bytes(comm)}")
    print(f"    Hurdle rate:  {format_bytes(hurdle_comm)}")
    print(f"    Performance:  {comm_pct_str}  {comm_status}")
    print()

    print(f"{BOLD}{'═' * 70}{RESET}")

    if all_passed:
        print(f"{GREEN}{BOLD}  ✓ ALL HURDLE RATES PASSED!{RESET}")
        print(
            f"{GREEN}  Your strategy meets the minimum performance thresholds.{RESET}"
        )
    else:
        print(f"{YELLOW}{BOLD}  ⚠ SOME HURDLE RATES NOT MET{RESET}")
        print(f"{YELLOW}  Consider optimizing your strategy before submission.{RESET}")

    print(f"{BOLD}{'═' * 70}{RESET}")
    print()

    # Raw metrics JSON for reference
    print(f"{DIM}Raw metrics (JSON):{RESET}")
    print(f"{DIM}{json.dumps(metrics, indent=2)}{RESET}")
    print()


def run_evaluation(strategy_path, num_nodes, max_steps, model_size, dataset, device):
    """Run the evaluation and return metrics."""

    train_dataset, _ = get_dataset(
        dataset,
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * num_nodes,
    )
    val_dataset, _ = get_dataset(
        dataset, block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )
    model = GPT(GPTConfig.gpt2_size_map(model_size))
    trainer = Trainer(model, train_dataset, val_dataset, device=device)

    metrics_out = trainer.fit(
        max_steps=max_steps,
        strategy=strategy_path,
        num_epochs=1,
        num_nodes=num_nodes,
        device=device,
        batch_size=256,
        minibatch_size=16,
        shuffle=False,
        val_size=256,
        val_interval=10,
    )

    result = {
        "throughput": int(metrics_out.get("tokens_per_sec", 0)),
        "loss": float(metrics_out.get("eval_loss", 0.0)),
        "communication": int(metrics_out.get("comm_bytes_total", 0)),
    }

    return result, model, trainer, train_dataset, val_dataset


def cleanup(
    model=None, trainer=None, train_dataset=None, val_dataset=None, device="cuda"
):
    """Clean up resources after evaluation."""
    try:
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        if train_dataset is not None:
            del train_dataset
        if val_dataset is not None:
            del val_dataset
    except:
        pass
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main_sandbox():
    """
    Sandbox mode: Run evaluation inside Docker container.
    Outputs JSON to stdout for evaluator.py to parse.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    # Configuration from environment variables
    num_nodes = int(os.getenv("NUM_NODES", "4"))
    max_steps = int(os.getenv("MAX_STEPS", "100"))
    model_size = os.getenv("MODEL_SIZE", "medium")
    dataset = os.getenv("DATASET", "owt")
    device = os.getenv("DEVICE", "cuda")
    strategy_path = os.getenv("STRATEGY_PATH", "/sandbox/strategy.py")

    logger.info(f"MAX_STEPS: {max_steps}")
    logger.info(f"NUM_NODES: {num_nodes}")

    model, trainer, train_dataset, val_dataset = None, None, None, None

    try:
        result, model, trainer, train_dataset, val_dataset = run_evaluation(
            strategy_path, num_nodes, max_steps, model_size, dataset, device
        )

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
        cleanup(model, trainer, train_dataset, val_dataset, device)

def main_local(args):
    """
    Local testing mode: Run evaluation on local strategy file.
    Displays formatted results with hurdle rate comparison.
    """
    # Set up logging (less verbose for local mode)
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    # Validate strategy file exists
    if not os.path.isfile(args.strategy):
        print(f"\033[91mError: Strategy file not found: {args.strategy}\033[0m")
        sys.exit(1)

    strategy_path = os.path.abspath(args.strategy)

    print()
    print(f"\033[1m\033[96m  DisTrOpZ Local Strategy Tester\033[0m")
    print(f"\033[2m  Testing: {strategy_path}\033[0m")
    print()

    config = {
        "num_nodes": args.num_nodes,
        "max_steps": args.max_steps,
        "model_size": args.model_size,
        "dataset": args.dataset,
        "device": args.device,
    }

    model, trainer, train_dataset, val_dataset = None, None, None, None

    try:
        print(f"\033[2mStarting evaluation...\033[0m")
        print()

        result, model, trainer, train_dataset, val_dataset = run_evaluation(
            strategy_path,
            args.num_nodes,
            args.max_steps,
            args.model_size,
            args.dataset,
            args.device,
        )

        # Build hurdle rates from CLI arguments
        hurdle_rates = {
            "throughput": args.hurdle_throughput,
            "loss": args.hurdle_loss,
            "communication": args.hurdle_comm,
        }

        # Print formatted results
        print_local_results(result, hurdle_rates, config)

        # Return exit code based on whether all hurdles passed
        # (hurdle of 0 means no minimum/maximum enforced for that metric)
        throughput_pass = (
            hurdle_rates["throughput"] == 0
            or result["throughput"] >= hurdle_rates["throughput"]
        )
        loss_pass = hurdle_rates["loss"] == 0 or result["loss"] <= hurdle_rates["loss"]
        comm_pass = (
            hurdle_rates["communication"] == 0
            or result["communication"] <= hurdle_rates["communication"]
        )

        if args.strict and not (throughput_pass and loss_pass and comm_pass):
            sys.exit(1)

    except Exception as e:
        print(f"\033[91mError during evaluation: {e}\033[0m")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    finally:
        cleanup(model, trainer, train_dataset, val_dataset, args.device)


def main():
    """Main entry point - detect mode and dispatch."""
    parser = argparse.ArgumentParser(
        description="DisTrOpZ Strategy Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test your local strategy against default hurdle rates:
  python evaluation_sandbox.py --local --strategy ./my_strategy.py
  
  # Test with specific configuration:
  python evaluation_sandbox.py --local --strategy ./my_strategy.py \\
      --num-nodes 4 --max-steps 1000 --model-size large --dataset owt
  
  # Custom hurdle rates:
  python evaluation_sandbox.py --local --strategy ./my_strategy.py \\
      --hurdle-loss 5.0 --hurdle-comm 50000000000 --hurdle-throughput 10000
  
  # Run in strict mode (exit code 1 if hurdles not met):
  python evaluation_sandbox.py --local --strategy ./my_strategy.py --strict
  
  # Verbose mode with debug output:
  python evaluation_sandbox.py --local --strategy ./my_strategy.py --verbose

Default Hurdle Rates:
  - Min Throughput: 211,876 tokens/sec (higher is better)
  - Max Loss: 6.35 (lower is better)
  - Max Communication: 70,264,572,848 bytes (lower is better)
""",
    )

    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        help="Run in local testing mode (instead of sandbox mode)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        help="Path to your strategy file (required for --local mode)",
    )
    parser.add_argument(
        "--num_nodes",
        "-n",
        type=int,
        default=int(os.getenv("NUM_NODES", "4")),
        help="Number of simulated nodes/GPUs (default: 4)",
    )
    parser.add_argument(
        "--max_steps",
        "-m",
        type=int,
        default=int(os.getenv("MAX_STEPS", "100")),
        help="Maximum training steps (default: 100)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=os.getenv("MODEL_SIZE", "medium"),
        choices=["small", "medium", "large", "xl"],
        help="GPT model size (default: medium)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("DATASET", "owt"),
        help="Dataset to use (default: owt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cuda"),
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: cuda)",
    )
    # Hurdle rate arguments
    parser.add_argument(
        "--hurdle-throughput",
        type=int,
        default=DEFAULT_HURDLE_RATES["throughput"],
        metavar="N",
        help=f"Min throughput hurdle in tokens/sec (default: {DEFAULT_HURDLE_RATES['throughput']:,})",
    )
    parser.add_argument(
        "--hurdle-loss",
        type=float,
        default=DEFAULT_HURDLE_RATES["loss"],
        metavar="N",
        help=f"Max loss hurdle (default: {DEFAULT_HURDLE_RATES['loss']})",
    )
    parser.add_argument(
        "--hurdle-comm",
        type=int,
        default=DEFAULT_HURDLE_RATES["communication"],
        metavar="N",
        help=f"Max communication hurdle in bytes (default: {DEFAULT_HURDLE_RATES['communication']:,})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any hurdle rate is not met",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug output"
    )

    args = parser.parse_args()

    # Determine mode
    if args.local:
        if not args.strategy:
            parser.error("--strategy is required when using --local mode")
        main_local(args)
    else:
        # Sandbox mode - ignore CLI args, use environment variables
        main_sandbox()


if __name__ == "__main__":
    main()
