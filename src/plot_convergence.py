from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


D3QN_FILE = "results/d3qn_medium_training_history.csv"
PPO_FILE = "results/ppo_medium_training_history.csv"

OUTPUT_FILE = "results/d3qn_vs_ppo_convergence.png"


def main() -> None:

    print("=" * 70)
    print("D3QN VS PPO TRAINING CONVERGENCE")
    print("=" * 70)

    if not os.path.exists(D3QN_FILE):
        raise FileNotFoundError(
            f"Missing file: {D3QN_FILE}"
        )

    if not os.path.exists(PPO_FILE):
        raise FileNotFoundError(
            f"Missing file: {PPO_FILE}"
        )

    d3qn = pd.read_csv(D3QN_FILE)
    ppo = pd.read_csv(PPO_FILE)

    print(
        "D3QN episodes:",
        len(d3qn)
    )

    print(
        "PPO episodes:",
        len(ppo)
    )

    if "episode" not in d3qn.columns:
        d3qn["episode"] = range(
            1,
            len(d3qn) + 1
        )

    if "episode" not in ppo.columns:
        ppo["episode"] = range(
            1,
            len(ppo) + 1
        )

    if "reward" not in d3qn.columns:
        raise ValueError(
            "D3QN CSV has no reward column."
        )

    if "reward" not in ppo.columns:
        raise ValueError(
            "PPO CSV has no reward column."
        )

    # 20-episode moving average
    d3qn["moving_average"] = (
        d3qn["reward"]
        .rolling(
            window=20,
            min_periods=1
        )
        .mean()
    )

    ppo["moving_average"] = (
        ppo["reward"]
        .rolling(
            window=20,
            min_periods=1
        )
        .mean()
    )

    plt.figure(
        figsize=(10, 6)
    )

    # D3QN raw reward
    plt.plot(
        d3qn["episode"],
        d3qn["reward"],
        color="tab:blue",
        alpha=0.20,
        linewidth=1
    )

    # D3QN convergence
    plt.plot(
        d3qn["episode"],
        d3qn["moving_average"],
        color="tab:blue",
        linewidth=2.5,
        label="D3QN"
    )

    # PPO raw reward
    plt.plot(
        ppo["episode"],
        ppo["reward"],
        color="tab:orange",
        alpha=0.20,
        linewidth=1
    )

    # PPO convergence
    plt.plot(
        ppo["episode"],
        ppo["moving_average"],
        color="tab:orange",
        linewidth=2.5,
        label="PPO"
    )

    plt.xlabel(
        "Training Episode"
    )

    plt.ylabel(
        "Episode Reward"
    )

    plt.title(
        "Training Reward Convergence: D3QN vs PPO"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    os.makedirs(
        "results",
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("=" * 70)
    print("CONVERGENCE PLOT CREATED")
    print("=" * 70)

    print(
        "Saved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()