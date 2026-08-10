from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOW_CSV = "results/d3q_low_final_10_runs.csv"
HIGH_CSV = "results/d3q_high_final_10_runs.csv"
OUTPUT_FOLDER = "figures/d3q_low_vs_high"

SLICES = ["eMBB", "URLLC1", "URLLC2", "BE1"]


def load_results():
    low_data = pd.read_csv(LOW_CSV)
    high_data = pd.read_csv(HIGH_CSV)

    return low_data, high_data


def create_slice_graph(
    low_data,
    high_data,
    metric_suffix,
    ylabel,
    title,
    filename,
):
    low_values = [
        low_data[f"{slice_name}_{metric_suffix}"].mean()
        for slice_name in SLICES
    ]

    high_values = [
        high_data[f"{slice_name}_{metric_suffix}"].mean()
        for slice_name in SLICES
    ]

    x = np.arange(len(SLICES))
    width = 0.35

    figure, axis = plt.subplots(figsize=(10, 6))

    low_bars = axis.bar(
        x - width / 2,
        low_values,
        width,
        label="Low traffic",
    )

    high_bars = axis.bar(
        x + width / 2,
        high_values,
        width,
        label="High traffic",
    )

    axis.set_title(title)
    axis.set_xlabel("Network slice")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x)
    axis.set_xticklabels(SLICES)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    axis.bar_label(
        low_bars,
        fmt="%.2f",
        padding=3,
    )

    axis.bar_label(
        high_bars,
        fmt="%.2f",
        padding=3,
    )

    figure.tight_layout()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def create_reward_graph(
    low_data,
    high_data,
):
    load_names = ["Low traffic", "High traffic"]

    rewards = [
        low_data["total_reward"].mean(),
        high_data["total_reward"].mean(),
    ]

    figure, axis = plt.subplots(figsize=(7, 6))

    bars = axis.bar(
        load_names,
        rewards,
    )

    axis.set_title(
        "D3QN Average Reward: Low vs High Traffic"
    )

    axis.set_ylabel("Average total reward")
    axis.grid(axis="y", alpha=0.3)

    axis.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
    )

    figure.tight_layout()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "01_reward_low_vs_high.png",
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main():
    if not os.path.exists(LOW_CSV):
        raise FileNotFoundError(
            f"Low-traffic CSV not found: {LOW_CSV}"
        )

    if not os.path.exists(HIGH_CSV):
        raise FileNotFoundError(
            f"High-traffic CSV not found: {HIGH_CSV}"
        )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True,
    )

    low_data, high_data = load_results()

    create_reward_graph(
        low_data,
        high_data,
    )

    create_slice_graph(
        low_data=low_data,
        high_data=high_data,
        metric_suffix="throughput_percent",
        ylabel="Average throughput (%)",
        title="D3QN Throughput: Low vs High Traffic",
        filename="02_throughput_low_vs_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        high_data=high_data,
        metric_suffix="latency_ms",
        ylabel="Average latency (ms)",
        title="D3QN Latency: Low vs High Traffic",
        filename="03_latency_low_vs_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        high_data=high_data,
        metric_suffix="jitter_ms",
        ylabel="Average jitter (ms)",
        title="D3QN Jitter: Low vs High Traffic",
        filename="04_jitter_low_vs_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        high_data=high_data,
        metric_suffix="plr_percent",
        ylabel="Packet loss ratio (%)",
        title="D3QN Packet Loss Ratio: Low vs High Traffic",
        filename="05_plr_low_vs_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        high_data=high_data,
        metric_suffix="allocation_percent",
        ylabel="Resource allocation (%)",
        title="D3QN Resource Allocation: Low vs High Traffic",
        filename="06_allocation_low_vs_high.png",
    )

    print(
        "\nAll Low vs High traffic graphs were created successfully."
    )


if __name__ == "__main__":
    main()