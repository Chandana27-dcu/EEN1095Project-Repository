from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

methods = ["Baseline", "PPO", "D3QN"]
slices = ["eMBB", "URLLC1", "URLLC2", "BE1"]

# Current evaluation results.
# Baseline values use Baseline Action 3 because it achieved
# the highest total baseline reward.
average_rewards = [385.7015 / 500, 0.8451, 0.9019]

throughput = {
    "Baseline": [3.1300, 98.9057, 100.0000, 100.0000],
    "PPO": [58.0600, 100.0000, 100.0000, 100.0000],
    "D3QN": [62.5142, 99.6408, 100.0000, 99.7520],
}

latency = {
    "Baseline": [94.7561, 1.0010, 1.0000, 1.0000],
    "PPO": [52.0333, 1.0000, 1.0000, 1.0000],
    "D3QN": [43.8648, 1.0001, 1.0000, 1.0002],
}

jitter = {
    "Baseline": [43.6459, 0.0315, 0.0000, 0.0000],
    "PPO": [12.7273, 0.0000, 0.0000, 0.0000],
    "D3QN": [18.6436, 0.0033, 0.0000, 0.0044],
}

plr = {
    "Baseline": [83.8472, 0.0000, 0.0000, 0.0000],
    "PPO": [39.5109, 0.0000, 0.0000, 0.0000],
    "D3QN": [3.1434, 0.0000, 0.0000, 0.0000],
}

allocation = {
    "Baseline": [5.0000, 5.0000, 35.0000, 55.0000],
    "PPO": [25.2400, 20.3100, 29.0000, 25.4500],
    "D3QN": [21.5000, 25.5000, 25.5000, 27.5000],
}


def add_bar_labels(axis, bars, decimals=1, minimum_value=0.05):
    """Add labels only to values large enough to be useful."""
    for bar in bars:
        value = bar.get_height()

        if value < minimum_value:
            continue

        axis.annotate(
            f"{value:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_reward():
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
    positions,
    data[method],
    width,
    label=method,
    edgecolor="black",
    linewidth=0.5,
)

    ax.set_title(
        "Average Reward Comparison under Medium Traffic Load",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Resource Allocation Method",
        fontsize=14,
    )
    ax.set_ylabel(
        "Average Reward per Step",
        fontsize=14,
    )
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.2)

    add_bar_labels(
        ax,
        bars,
        decimals=3,
        minimum_value=0,
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "01_average_reward_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_grouped_metric(
    data,
    ylabel,
    title,
    filename,
    decimals=1,
    minimum_label_value=0.05,
):
    x = np.arange(len(slices))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, method in enumerate(methods):
        positions = x + (index - 1) * width

        bars = ax.bar(
            positions,
            data[method],
            width,
            label=method,
        )

        add_bar_labels(
            ax,
            bars,
            decimals=decimals,
            minimum_value=minimum_label_value,
        )

    ax.set_title(
    title,
    fontsize=16,
    fontweight="bold",
    pad=15,
)
    ax.set_xlabel(
        "Network Slice",
        fontsize=14,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(slices)
    ax.tick_params(axis="both", labelsize=12)
   ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    fontsize=11,
)
    ax.grid(axis="y", alpha=0.3)

    maximum = max(max(values) for values in data.values())
    ax.set_ylim(0, maximum * 1.18 if maximum > 0 else 1)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    plot_reward()

    plot_grouped_metric(
        throughput,
        "Average Throughput (%)",
        "Throughput Comparison under Medium Traffic Load",
        "02_throughput_comparison.png",
        decimals=1,
        minimum_label_value=0.05,
    )

    plot_grouped_metric(
        latency,
        "Average Latency (ms)",
        "Latency Comparison under Medium Traffic Load",
        "03_latency_comparison.png",
        decimals=1,
        minimum_label_value=2.0,
    )

    plot_grouped_metric(
        jitter,
        "Average Jitter (ms)",
        "Jitter Comparison under Medium Traffic Load",
        "04_jitter_comparison.png",
        decimals=1,
        minimum_label_value=1.0,
    )

    plot_grouped_metric(
        plr,
        "Packet Loss Ratio (%)",
        "Packet Loss Ratio Comparison under Medium Traffic Load",
        "05_plr_comparison.png",
        decimals=1,
        minimum_label_value=0.1,
    )

    plot_grouped_metric(
        allocation,
        "Resource Allocation (%)",
        "Resource Allocation Comparison under Medium Traffic Load",
        "06_resource_allocation_comparison.png",
        decimals=1,
        minimum_label_value=0.05,
    )

    print("All six comparison graphs were created successfully:")
    print("1. figures/01_average_reward_comparison.png")
    print("2. figures/02_throughput_comparison.png")
    print("3. figures/03_latency_comparison.png")
    print("4. figures/04_jitter_comparison.png")
    print("5. figures/05_plr_comparison.png")
    print("6. figures/06_resource_allocation_comparison.png")


if __name__ == "__main__":
    main()