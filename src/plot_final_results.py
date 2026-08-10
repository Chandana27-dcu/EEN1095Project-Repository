from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "results/final_all_methods_summary.csv"

OUTPUT_DIR = "results/final_plots"


# ============================================================
# CONSISTENT METHOD NAMES AND COLOURS
# ============================================================

METHODS = [
    "D3QN",
    "PPO",
    "Baseline",
]


METHOD_COLORS = {
    "D3QN": "tab:blue",
    "PPO": "tab:orange",
    "Baseline": "tab:green",
}


METHOD_LABELS = {
    "D3QN": "D3QN",
    "PPO": "PPO",
    "Baseline": "Static Equal Allocation (SEA)",
}


TRAFFIC_ORDER = [
    "low",
    "medium",
    "high",
]


TRAFFIC_LABELS = [
    "Low",
    "Medium",
    "High",
]


# ============================================================
# GENERIC PLOTTING FUNCTION
# ============================================================

def plot_metric(
    data: pd.DataFrame,
    metric_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_filename: str,
) -> None:
    """
    Create one Low/Medium/High comparison plot.

    The same colour is used for each method in every graph:
        D3QN     -> Blue
        PPO      -> Orange
        SEA      -> Green
    """

    plt.figure(
        figsize=(9, 6)
    )

    for method in METHODS:

        method_data = (
            data[
                data["method"] == method
            ]
            .set_index("traffic_load")
            .reindex(TRAFFIC_ORDER)
        )

        if method_data.empty:
            raise ValueError(
                f"No data found for method: {method}"
            )

        mean_values = method_data[
            metric_column
        ].to_numpy()

        std_values = method_data[
            std_column
        ].to_numpy()

        plt.errorbar(
            TRAFFIC_LABELS,
            mean_values,
            yerr=std_values,
            marker="o",
            markersize=7,
            linewidth=2.5,
            capsize=4,
            color=METHOD_COLORS[
                method
            ],
            label=METHOD_LABELS[
                method
            ],
        )

    plt.xlabel(
        "Traffic Load",
        fontsize=12,
    )

    plt.ylabel(
        ylabel,
        fontsize=12,
    )

    plt.title(
        title,
        fontsize=14,
    )

    plt.legend(
        fontsize=10
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 75)
    print("FINAL D3QN / PPO / SEA COMPARISON PLOTS")
    print("=" * 75)

    # --------------------------------------------------------
    # Check input CSV
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):
        raise FileNotFoundError(
            f"Cannot find input file: "
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load final evaluation summary
    # --------------------------------------------------------

    data = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Input file:",
        INPUT_FILE
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "Methods:",
        data[
            "method"
        ].unique().tolist()
    )

    print(
        "Traffic loads:",
        data[
            "traffic_load"
        ].unique().tolist()
    )

    print()

    # ========================================================
    # 1. REWARD
    # ========================================================

    plot_metric(
        data=data,

        metric_column=
            "reward_mean",

        std_column=
            "reward_std",

        ylabel=
            "Average Episode Reward",

        title=
            "Average Reward vs Traffic Load",

        output_filename=
            "01_reward_vs_traffic.png",
    )

    # ========================================================
    # 2. THROUGHPUT
    # ========================================================

    plot_metric(
        data=data,

        metric_column=
            "throughput_mean",

        std_column=
            "throughput_std",

        ylabel=
            "Average Throughput (%)",

        title=
            "Average Throughput vs Traffic Load",

        output_filename=
            "02_throughput_vs_traffic.png",
    )

    # ========================================================
    # 3. LATENCY
    # ========================================================

    plot_metric(
        data=data,

        metric_column=
            "latency_mean",

        std_column=
            "latency_std",

        ylabel=
            "Average Latency (ms)",

        title=
            "Average Latency vs Traffic Load",

        output_filename=
            "03_latency_vs_traffic.png",
    )

    # ========================================================
    # 4. JITTER
    # ========================================================

    plot_metric(
        data=data,

        metric_column=
            "jitter_mean",

        std_column=
            "jitter_std",

        ylabel=
            "Average Jitter (ms)",

        title=
            "Average Jitter vs Traffic Load",

        output_filename=
            "04_jitter_vs_traffic.png",
    )

    # ========================================================
    # 5. PACKET LOSS RATIO
    # ========================================================

    plot_metric(
        data=data,

        metric_column=
            "plr_mean",

        std_column=
            "plr_std",

        ylabel=
            "Average Packet Loss Ratio (%)",

        title=
            "Average PLR vs Traffic Load",

        output_filename=
            "05_plr_vs_traffic.png",
    )

    # ========================================================
    # COMPLETED
    # ========================================================

    print()
    print("=" * 75)
    print("ALL FINAL COMPARISON PLOTS CREATED")
    print("=" * 75)

    print(
        "Colour coding:"
    )

    print(
        "D3QN = Blue"
    )

    print(
        "PPO = Orange"
    )

    print(
        "Static Equal Allocation (SEA) = Green"
    )

    print()

    print(
        f"Plots saved in: "
        f"{OUTPUT_DIR}"
    )

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()