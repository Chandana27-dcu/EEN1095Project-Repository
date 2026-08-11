from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

RESULT_DIR = "results/epoch_analysis"

D3QN_TRAINING_FILE = os.path.join(
    RESULT_DIR,
    "d3qn_training_per_episode.csv",
)

PPO_TRAINING_FILE = os.path.join(
    RESULT_DIR,
    "ppo_training_per_episode.csv",
)

TESTING_FILE = os.path.join(
    RESULT_DIR,
    "checkpoint_testing_summary.csv",
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MAX_EPISODE = 200

MOVING_AVERAGE_WINDOW = 20

METHODS = [
    "D3QN",
    "PPO",
    "SEA",
]


# ============================================================
# CONSISTENT COLOURS
# ============================================================

METHOD_COLORS = {
    "D3QN": "tab:blue",
    "PPO": "tab:orange",
    "SEA": "tab:green",
}


METHOD_LABELS = {
    "D3QN": "D3QN",
    "PPO": "PPO",
    "SEA": "Static Equal Allocation (SEA)",
}


# ============================================================
# CHECK INPUT FILES
# ============================================================

def check_files() -> None:

    required_files = [
        D3QN_TRAINING_FILE,
        PPO_TRAINING_FILE,
        TESTING_FILE,
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Missing required file: "
                f"{file_path}"
            )


# ============================================================
# 1. TRAINING REWARD VS EPISODE
# ============================================================

def plot_training_reward() -> None:

    d3qn = pd.read_csv(
        D3QN_TRAINING_FILE
    )

    ppo = pd.read_csv(
        PPO_TRAINING_FILE
    )

    # Keep only Episodes 1-200.
    d3qn = d3qn[
        (
            d3qn["episode"] >= 1
        )
        &
        (
            d3qn["episode"]
            <= MAX_EPISODE
        )
    ].copy()

    ppo = ppo[
        (
            ppo["episode"] >= 1
        )
        &
        (
            ppo["episode"]
            <= MAX_EPISODE
        )
    ].copy()

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    d3qn[
        "moving_average"
    ] = (
        d3qn["reward"]
        .rolling(
            window=
                MOVING_AVERAGE_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    ppo[
        "moving_average"
    ] = (
        ppo["reward"]
        .rolling(
            window=
                MOVING_AVERAGE_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    # D3QN raw reward
    plt.plot(
        d3qn["episode"],
        d3qn["reward"],
        color=
            METHOD_COLORS[
                "D3QN"
            ],
        alpha=0.20,
        linewidth=1.0,
    )

    # D3QN moving average
    plt.plot(
        d3qn["episode"],
        d3qn[
            "moving_average"
        ],
        color=
            METHOD_COLORS[
                "D3QN"
            ],
        linewidth=2.5,
        label=
            (
                "D3QN "
                f"({MOVING_AVERAGE_WINDOW}-episode "
                "moving average)"
            ),
    )

    # PPO raw reward
    plt.plot(
        ppo["episode"],
        ppo["reward"],
        color=
            METHOD_COLORS[
                "PPO"
            ],
        alpha=0.20,
        linewidth=1.0,
    )

    # PPO moving average
    plt.plot(
        ppo["episode"],
        ppo[
            "moving_average"
        ],
        color=
            METHOD_COLORS[
                "PPO"
            ],
        linewidth=2.5,
        label=
            (
                "PPO "
                f"({MOVING_AVERAGE_WINDOW}-episode "
                "moving average)"
            ),
    )

    plt.xlabel(
        "Checkpoint Episode",
        fontsize=12,
    )

    plt.ylabel(
        "Episode Reward",
        fontsize=12,
    )

    plt.title(
        "Training Reward vs Episode",
        fontsize=14,
    )

    plt.xlim(
        1,
        MAX_EPISODE,
    )

    plt.legend(
        fontsize=10
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR,
        "01_training_reward_vs_episode.png",
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
# GENERIC TESTING PLOT
# ============================================================

def plot_testing_metric(
    data: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_filename: str,
) -> None:

    plt.figure(
        figsize=(10, 6)
    )

    for method in METHODS:

        method_data = (
            data[
                data["method"]
                == method
            ]
            .sort_values(
                "episode"
            )
        )

        if method_data.empty:

            raise ValueError(
                f"No testing data "
                f"found for {method}."
            )

        plt.errorbar(
            method_data[
                "episode"
            ],

            method_data[
                mean_column
            ],

            yerr=
                method_data[
                    std_column
                ],

            marker="o",

            markersize=5,

            linewidth=2.2,

            capsize=3,

            color=
                METHOD_COLORS[
                    method
                ],

            label=
                METHOD_LABELS[
                    method
                ],
        )

    plt.xlabel(
        "Training Episode",
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

    plt.xticks(
        range(
            10,
            201,
            20,
        )
    )

    plt.xlim(
        5,
        205,
    )

    plt.legend(
        fontsize=10
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR,
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
    print(
        "TRAINING / TESTING "
        "PER-EPISODE PLOTS"
    )
    print("=" * 75)

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    check_files()

    # ========================================================
    # TRAINING PLOT
    # ========================================================

    plot_training_reward()

    # ========================================================
    # LOAD TESTING SUMMARY
    # ========================================================

    testing_data = pd.read_csv(
        TESTING_FILE
    )

    testing_data = testing_data[
        testing_data["episode"]
        <= MAX_EPISODE
    ].copy()

    print(
        f"Testing summary rows: "
        f"{len(testing_data)}"
    )

    # ========================================================
    # 2. TESTING REWARD
    # ========================================================

    plot_testing_metric(

        data=
            testing_data,

        mean_column=
            "reward_mean",

        std_column=
            "reward_std",

        ylabel=
            "Mean Test Episode Reward",

        title=
            (
                "Testing Reward vs "
                "Training Episode "
                "(Medium Traffic)"
            ),

        output_filename=
            "02_testing_reward_vs_episode.png",
    )

    # ========================================================
    # 3. TESTING THROUGHPUT
    # ========================================================

    plot_testing_metric(

        data=
            testing_data,

        mean_column=
            "throughput_mean",

        std_column=
            "throughput_std",

        ylabel=
            "Mean Throughput (%)",

        title=
            (
                "Testing Throughput vs "
                "Training Episode "
                "(Medium Traffic)"
            ),

        output_filename=
            "03_testing_throughput_vs_episode.png",
    )

    # ========================================================
    # 4. TESTING LATENCY
    # ========================================================

    plot_testing_metric(

        data=
            testing_data,

        mean_column=
            "latency_mean",

        std_column=
            "latency_std",

        ylabel=
            "Mean Latency (ms)",

        title=
            (
                "Testing Latency vs "
                "Training Episode "
                "(Medium Traffic)"
            ),

        output_filename=
            "04_testing_latency_vs_episode.png",
    )

    # ========================================================
    # 5. TESTING JITTER
    # ========================================================

    plot_testing_metric(

        data=
            testing_data,

        mean_column=
            "jitter_mean",

        std_column=
            "jitter_std",

        ylabel=
            "Mean Jitter (ms)",

        title=
            (
                "Testing Jitter vs "
                "Training Episode "
                "(Medium Traffic)"
            ),

        output_filename=
            "05_testing_jitter_vs_episode.png",
    )

    # ========================================================
    # 6. TESTING PACKET LOSS RATIO
    # ========================================================

    plot_testing_metric(

        data=
            testing_data,

        mean_column=
            "plr_mean",

        std_column=
            "plr_std",

        ylabel=
            "Mean Packet Loss Ratio (%)",

        title=
            (
                "Testing PLR vs "
                "Training Episode "
                "(Medium Traffic)"
            ),

        output_filename=
            "06_testing_plr_vs_episode.png",
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 75)
    print(
        "ALL PER-EPISODE "
        "PLOTS CREATED"
    )
    print("=" * 75)

    print(
        "D3QN = Blue"
    )

    print(
        "PPO = Orange"
    )

    print(
        "SEA = Green"
    )

    print()

    print(
        f"Plots saved in: "
        f"{RESULT_DIR}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()