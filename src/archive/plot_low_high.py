from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv
from src.network_d3q import DuelingQNetwork


MODEL_PATH = "models/d3q_network_slicing.pth"
OUTPUT_FOLDER = "results/low_high_graphs"


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    network = DuelingQNetwork(
        state_size=CONFIG["STATE_SIZE"],
        action_size=CONFIG["NUMBER_OF_ACTIONS"],
        hidden_size=CONFIG["HIDDEN_SIZE"],
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        network.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        network.load_state_dict(checkpoint)

    network.eval()

    return network


def select_action(network, state, device):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state_tensor)

    return int(
        torch.argmax(q_values, dim=1).item()
    )


def run_episode(
    network,
    traffic_load,
    seed,
    device,
):
    env = NetworkSlicingD3QEnv(
        traffic_load=traffic_load
    )

    state, _ = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = select_action(
            network,
            state,
            device,
        )

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

    metrics = env.get_episode_metrics()

    return total_reward, metrics


def evaluate_load(
    network,
    traffic_load,
    number_of_runs,
    device,
):
    slices = CONFIG["SLICES"]

    throughput = {
        name: [] for name in slices
    }

    latency = {
        name: [] for name in slices
    }

    jitter = {
        name: [] for name in slices
    }

    plr = {
        name: [] for name in slices
    }

    rewards = []

    for run in range(number_of_runs):
        seed = (
            CONFIG["EVALUATION"]["base_seed"]
            + run
        )

        reward, metrics = run_episode(
            network=network,
            traffic_load=traffic_load,
            seed=seed,
            device=device,
        )

        rewards.append(reward)

        for slice_name in slices:
            throughput[slice_name].append(
                metrics[slice_name][
                    "average_throughput_percent"
                ]
            )

            latency[slice_name].append(
                metrics[slice_name][
                    "average_latency_ms"
                ]
            )

            jitter[slice_name].append(
                metrics[slice_name][
                    "jitter_ms"
                ]
            )

            plr[slice_name].append(
                metrics[slice_name][
                    "plr_percent"
                ]
            )

        print(
            f"{traffic_load} traffic: "
            f"run {run + 1}/{number_of_runs}, "
            f"reward = {reward:.3f}"
        )

    return {
        "reward": float(np.mean(rewards)),

        "throughput": {
            name: float(np.mean(values))
            for name, values in throughput.items()
        },

        "latency": {
            name: float(np.mean(values))
            for name, values in latency.items()
        },

        "jitter": {
            name: float(np.mean(values))
            for name, values in jitter.items()
        },

        "plr": {
            name: float(np.mean(values))
            for name, values in plr.items()
        },
    }


def create_graph(
    low_results,
    high_results,
    metric,
    ylabel,
    title,
    filename,
):
    slices = CONFIG["SLICES"]

    low_values = [
        low_results[metric][name]
        for name in slices
    ]

    high_values = [
        high_results[metric][name]
        for name in slices
    ]

    x = np.arange(len(slices))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        low_values,
        width,
        label="Low traffic",
    )

    plt.bar(
        x + width / 2,
        high_values,
        width,
        label="High traffic",
    )

    plt.xlabel("Network slice")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(
        x,
        slices,
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Graph saved: {output_path}")


def main():
    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    network = load_model(device)

    number_of_runs = CONFIG[
        "EVALUATION"
    ]["number_of_runs"]

    print("\nEvaluating low traffic...")

    low_results = evaluate_load(
        network=network,
        traffic_load="low",
        number_of_runs=number_of_runs,
        device=device,
    )

    print("\nEvaluating high traffic...")

    high_results = evaluate_load(
        network=network,
        traffic_load="high",
        number_of_runs=number_of_runs,
        device=device,
    )

    create_graph(
        low_results,
        high_results,
        metric="throughput",
        ylabel="Average throughput (%)",
        title="D3QN Throughput: Low vs High Traffic",
        filename="throughput_low_vs_high.png",
    )

    create_graph(
        low_results,
        high_results,
        metric="latency",
        ylabel="Average latency (ms)",
        title="D3QN Latency: Low vs High Traffic",
        filename="latency_low_vs_high.png",
    )

    create_graph(
        low_results,
        high_results,
        metric="jitter",
        ylabel="Average jitter (ms)",
        title="D3QN Jitter: Low vs High Traffic",
        filename="jitter_low_vs_high.png",
    )

    create_graph(
        low_results,
        high_results,
        metric="plr",
        ylabel="Packet loss ratio (%)",
        title="D3QN PLR: Low vs High Traffic",
        filename="plr_low_vs_high.png",
    )

    print("\nAverage rewards:")
    print(
        f"Low traffic: "
        f"{low_results['reward']:.3f}"
    )
    print(
        f"High traffic: "
        f"{high_results['reward']:.3f}"
    )

    print(
        "\nAll graphs were saved inside: "
        f"{OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()