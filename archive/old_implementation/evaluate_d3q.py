from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv
from src.network_d3q import DuelingQNetwork


# Main model produced by train_d3q.py
MODEL_PATH = "models/d3q_network_slicing.pth"


def load_d3q_model(
    model_path: str,
    device: torch.device,
) -> DuelingQNetwork:
    """Create the D3QN network and load the trained weights."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nD3QN model was not found:\n{model_path}\n\n"
            "Check the models folder and update MODEL_PATH "
            "inside src/evaluate_d3q.py."
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    # Read dimensions from the checkpoint when available.
    state_size = int(
        checkpoint.get(
            "state_size",
            CONFIG["STATE_SIZE"],
        )
    )

    action_size = int(
        checkpoint.get(
            "action_size",
            CONFIG["NUMBER_OF_ACTIONS"],
        )
    )

    hidden_size = int(
        checkpoint.get(
            "hidden_size",
            CONFIG["HIDDEN_SIZE"],
        )
    )

    network = DuelingQNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
    ).to(device)

    # Your training script stores weights under model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]

    # This also supports a file containing only the raw state dictionary.
    elif isinstance(checkpoint, dict):
        model_state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported D3QN checkpoint format."
        )

    network.load_state_dict(model_state_dict)
    network.eval()

    return network


def select_greedy_action(
    state: np.ndarray,
    network: DuelingQNetwork,
    device: torch.device,
) -> int:
    """Select the action with the highest predicted Q-value."""

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

    return action


def run_evaluation(
    model_path: str = MODEL_PATH,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one complete D3QN evaluation episode."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("D3QN Network Slicing Evaluation")
    print("=" * 65)
    print(f"Device     : {device}")
    print(f"Model path : {model_path}")
    print(f"Seed       : {seed}")

    environment = NetworkSlicingD3QEnv()
    network = load_d3q_model(
        model_path=model_path,
        device=device,
    )

    state, _ = environment.reset(seed=seed)

    terminated = False
    truncated = False

    total_reward = 0.0
    number_of_steps = 0

    allocation_history = {
        slice_name: []
        for slice_name in environment.slices
    }

    action_history: list[int] = []
    final_info: dict[str, Any] = {}

    while not (terminated or truncated):
        action = select_greedy_action(
            state=state,
            network=network,
            device=device,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)

        total_reward += float(reward)
        number_of_steps += 1
        action_history.append(action)

        allocation = info.get(
            "allocation_percent",
            {},
        )

        for slice_name in environment.slices:
            allocation_history[slice_name].append(
                float(allocation.get(slice_name, 0.0))
            )

        state = next_state
        final_info = info

    average_reward = (
        total_reward / number_of_steps
        if number_of_steps > 0
        else 0.0
    )

    episode_metrics = final_info.get(
        "metrics",
        environment.get_episode_metrics(),
    )

    average_allocation = {
        slice_name: float(
            np.mean(allocation_history[slice_name])
        )
        if allocation_history[slice_name]
        else 0.0
        for slice_name in environment.slices
    }

    unique_actions, action_counts = np.unique(
        np.asarray(action_history, dtype=np.int64),
        return_counts=True,
    )

    most_used_action = (
        int(unique_actions[np.argmax(action_counts)])
        if len(unique_actions) > 0
        else -1
    )

    results = {
        "total_reward": total_reward,
        "average_reward": average_reward,
        "number_of_steps": number_of_steps,
        "average_allocation": average_allocation,
        "most_used_action": most_used_action,
        "metrics": episode_metrics,
    }

    environment.close()

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print the evaluation results clearly."""

    print("\n" + "=" * 65)
    print("D3QN Evaluation Result")
    print("=" * 65)

    print(
        f"Total Reward           : "
        f"{results['total_reward']:.4f}"
    )

    print(
        f"Average Reward per Step: "
        f"{results['average_reward']:.4f}"
    )

    print(
        f"Number of Steps        : "
        f"{results['number_of_steps']}"
    )

    print(
        f"Most Used Action Index : "
        f"{results['most_used_action']}"
    )

    print("\nAverage Resource Allocation")
    print("-" * 65)

    for slice_name, allocation in results[
        "average_allocation"
    ].items():
        print(
            f"{slice_name:<10}: "
            f"{allocation:>8.2f}%"
        )

    print("\nPer-Slice Metrics")
    print("-" * 65)

    metrics = results["metrics"]

    for slice_name, slice_metrics in metrics.items():
        throughput = float(
            slice_metrics.get(
                "average_throughput_percent",
                0.0,
            )
        )

        latency = float(
            slice_metrics.get(
                "average_latency_ms",
                0.0,
            )
        )

        jitter = float(
            slice_metrics.get(
                "jitter_ms",
                0.0,
            )
        )

        plr = float(
            slice_metrics.get(
                "plr_percent",
                0.0,
            )
        )

        queue_length = int(
            slice_metrics.get(
                "queue_length",
                0,
            )
        )

        throughput_bits = float(
            slice_metrics.get(
                "total_throughput_bits",
                0.0,
            )
        )

        final_allocation = float(
            slice_metrics.get(
                "final_allocation_percent",
                0.0,
            )
        )

        print(f"\n{slice_name}")
        print(
            f"  Average Throughput : "
            f"{throughput:.4f}%"
        )
        print(
            f"  Total Throughput   : "
            f"{throughput_bits:.2f} bits"
        )
        print(
            f"  Average Latency    : "
            f"{latency:.4f} ms"
        )
        print(
            f"  Jitter             : "
            f"{jitter:.4f} ms"
        )
        print(
            f"  Packet Loss Ratio  : "
            f"{plr:.4f}%"
        )
        print(
            f"  Final Queue Length : "
            f"{queue_length}"
        )
        print(
            f"  Final Allocation   : "
            f"{final_allocation:.2f}%"
        )

    print("\n" + "=" * 65)
    print("D3QN evaluation completed successfully.")
    print("=" * 65)


def main() -> None:
    try:
        results = run_evaluation()
        print_results(results)

    except FileNotFoundError as error:
        print(error)

        print("\nAvailable model files:")

        if os.path.isdir("models"):
            model_files = [
                filename
                for filename in os.listdir("models")
                if filename.endswith(".pth")
            ]

            if model_files:
                for filename in sorted(model_files):
                    print(f"  models/{filename}")
            else:
                print("  No .pth files found in models.")
        else:
            print("  The models folder does not exist.")

    except Exception as error:
        print("\nD3QN evaluation failed.")
        print(
            f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()