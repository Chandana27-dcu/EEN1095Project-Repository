from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOW_CSV = "results/d3q_low_final_10_runs.csv"
MEDIUM_CSV = "results/d3q_medium_load_10_runs.csv"
HIGH_CSV = "results/d3q_high_final_10_runs.csv"

OUTPUT_FOLDER = "figures/d3q_low_medium_high"

SLICES = ["eMBB", "URLLC1", "URLLC2", "BE1"]


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    low_data = pd.read_csv(LOW_CSV)
    medium_data = pd.read_csv(MEDIUM_CSV)
    high_data = pd.read_csv(HIGH_CSV)

    return low_data, medium_data, high_data


def check_required_files() -> None:
    files = [LOW_CSV, MEDIUM_CSV, HIGH_CSV]

    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"CSV file not found: {file_path}"
            )


def add_bar_labels(axis, bars) -> None:
    axis.bar_label(
        bars,
        fmt="%.2f",
        padding=3,
        fontsize=8,
    )


def create_reward_graph(
    low_data: pd.DataFrame,
    medium_data: pd.DataFrame,
    high_data: pd.DataFrame,
) -> None:
    traffic_levels = [
        "Low Traffic",
        "Medium Traffic",
        "High Traffic",
    ]

    reward_values = [
        low_data["total_reward"].mean(),
        medium_data["total_reward"].mean(),
        high_data["total_reward"].mean(),
    ]

    figure, axis = plt.subplots(figsize=(8, 6))

    bars = axis.bar(
        traffic_levels,
        reward_values,
        width=0.6,
    )

    axis.set_title(
        "D3QN Average Reward Under Different Traffic Loads",
        fontsize=14,
    )
    axis.set_xlabel("Traffic Load")
    axis.set_ylabel("Average Total Reward")
    axis.grid(axis="y", alpha=0.3)

    add_bar_labels(axis, bars)

    figure.tight_layout()

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "01_reward_low_medium_high.png",
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def create_slice_graph(
    low_data: pd.DataFrame,
    medium_data: pd.DataFrame,
    high_data: pd.DataFrame,
    metric_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    low_values = [
        low_data[f"{slice_name}_{metric_suffix}"].mean()
        for slice_name in SLICES
    ]

    medium_values = [
        medium_data[f"{slice_name}_{metric_suffix}"].mean()
        for slice_name in SLICES
    ]

    high_values = [
        high_data[f"{slice_name}_{metric_suffix}"].mean()
        for slice_name in SLICES
    ]

    x_positions = np.arange(len(SLICES))
    bar_width = 0.25

    figure, axis = plt.subplots(figsize=(11, 7))

    low_bars = axis.bar(
        x_positions - bar_width,
        low_values,
        bar_width,
        label="Low Traffic",
    )

    medium_bars = axis.bar(
        x_positions,
        medium_values,
        bar_width,
        label="Medium Traffic",
    )

    high_bars = axis.bar(
        x_positions + bar_width,
        high_values,
        bar_width,
        label="High Traffic",
    )

    axis.set_title(title, fontsize=14)
    axis.set_xlabel("Network Slice")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(SLICES)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    add_bar_labels(axis, low_bars)
    add_bar_labels(axis, medium_bars)
    add_bar_labels(axis, high_bars)

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


def create_summary_csv(
    low_data: pd.DataFrame,
    medium_data: pd.DataFrame,
    high_data: pd.DataFrame,
) -> None:
    rows = []

    traffic_datasets = {
        "Low": low_data,
        "Medium": medium_data,
        "High": high_data,
    }

    for traffic_level, dataset in traffic_datasets.items():
        row = {
            "traffic_load": traffic_level,
            "average_total_reward": dataset["total_reward"].mean(),
            "average_reward_per_step": dataset[
                "average_reward_per_step"
            ].mean(),
        }

        for slice_name in SLICES:
            row[f"{slice_name}_throughput_percent"] = dataset[
                f"{slice_name}_throughput_percent"
            ].mean()

            row[f"{slice_name}_latency_ms"] = dataset[
                f"{slice_name}_latency_ms"
            ].mean()

            row[f"{slice_name}_jitter_ms"] = dataset[
                f"{slice_name}_jitter_ms"
            ].mean()

            row[f"{slice_name}_plr_percent"] = dataset[
                f"{slice_name}_plr_percent"
            ].mean()

            row[f"{slice_name}_allocation_percent"] = dataset[
                f"{slice_name}_allocation_percent"
            ].mean()

        rows.append(row)

    summary_data = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "d3q_low_medium_high_summary.csv",
    )

    summary_data.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def main() -> None:
    check_required_files()

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True,
    )

    low_data, medium_data, high_data = load_results()

    create_reward_graph(
        low_data,
        medium_data,
        high_data,
    )

    create_slice_graph(
        low_data=low_data,
        medium_data=medium_data,
        high_data=high_data,
        metric_suffix="throughput_percent",
        ylabel="Average Throughput (%)",
        title="D3QN Throughput Under Low, Medium and High Traffic",
        filename="02_throughput_low_medium_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        medium_data=medium_data,
        high_data=high_data,
        metric_suffix="latency_ms",
        ylabel="Average Latency (ms)",
        title="D3QN Latency Under Low, Medium and High Traffic",
        filename="03_latency_low_medium_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        medium_data=medium_data,
        high_data=high_data,
        metric_suffix="jitter_ms",
        ylabel="Average Jitter (ms)",
        title="D3QN Jitter Under Low, Medium and High Traffic",
        filename="04_jitter_low_medium_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        medium_data=medium_data,
        high_data=high_data,
        metric_suffix="plr_percent",
        ylabel="Packet Loss Ratio (%)",
        title="D3QN Packet Loss Ratio Under Low, Medium and High Traffic",
        filename="05_plr_low_medium_high.png",
    )

    create_slice_graph(
        low_data=low_data,
        medium_data=medium_data,
        high_data=high_data,
        metric_suffix="allocation_percent",
        ylabel="Average Resource Allocation (%)",
        title="D3QN Resource Allocation Under Low, Medium and High Traffic",
        filename="06_allocation_low_medium_high.png",
    )

    create_summary_csv(
        low_data,
        medium_data,
        high_data,
    )

    print(
        "\nAll Low, Medium and High traffic graphs "
        "were created successfully."
    )


if __name__ == "__main__":
    main()