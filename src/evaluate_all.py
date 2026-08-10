from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from src.actions_common import ACTIONS
from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv
from src.network_d3q import DuelingQNetwork


# ============================================================
# FINAL TRAINED MODELS
# ============================================================

D3QN_MODEL_PATH = "models/d3qn_medium_load.pth"
PPO_MODEL_PATH = "models/ppo_medium_load.zip"


# ============================================================
# FINAL EVALUATION SETTINGS
# ============================================================

TRAFFIC_LOADS = [
    "low",
    "medium",
    "high",
]

SEEDS = list(
    range(42, 52)
)

BASELINE_ACTION = 95

DETAILED_RESULT_PATH = (
    "results/final_all_methods_detailed.csv"
)

RUN_SUMMARY_PATH = (
    "results/final_all_methods_run_summary.csv"
)

FINAL_SUMMARY_PATH = (
    "results/final_all_methods_summary.csv"
)


# ============================================================
# LOAD D3QN
# ============================================================

def load_d3qn(
    device: torch.device,
) -> DuelingQNetwork:
    """
    Load the final medium-trained D3QN model.
    """

    if not os.path.exists(
        D3QN_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Missing D3QN model: "
            f"{D3QN_MODEL_PATH}"
        )

    checkpoint = torch.load(
        D3QN_MODEL_PATH,
        map_location=device,
    )

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
            CONFIG["D3QN"][
                "hidden_size"
            ],
        )
    )

    network = DuelingQNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
    ).to(device)

    network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    network.eval()

    return network


# ============================================================
# D3QN ACTION
# ============================================================

def select_d3qn_action(
    state: np.ndarray,
    network: DuelingQNetwork,
    device: torch.device,
) -> int:
    """
    Select the greedy D3QN action.
    """

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = network(
            state_tensor
        )

    return int(
        torch.argmax(
            q_values,
            dim=1,
        ).item()
    )


# ============================================================
# CREATE EVALUATION ENVIRONMENT
# ============================================================

def create_environment(
    traffic_load: str,
    seed: int,
) -> tuple[
    NetworkSlicingD3QEnv,
    np.ndarray,
]:
    """
    Create one common evaluation environment.

    D3QN, PPO and the baseline all use this exact same
    environment implementation during final evaluation.
    """

    if traffic_load not in CONFIG[
        "TRAFFIC_SCENARIOS"
    ]:
        raise ValueError(
            f"Invalid traffic load: "
            f"{traffic_load}"
        )

    env = NetworkSlicingD3QEnv()

    # Override the evaluation traffic scenario without
    # retraining the medium-trained models.
    env.traffic_load = (
        traffic_load
    )

    env.traffic_config = (
        CONFIG[
            "TRAFFIC_SCENARIOS"
        ][traffic_load].copy()
    )

    state, _ = env.reset(
        seed=seed
    )

    return env, state


# ============================================================
# RUN ONE METHOD
# ============================================================

def run_episode(
    method: str,
    traffic_load: str,
    seed: int,
    d3qn_network:
        DuelingQNetwork,
    ppo_model: PPO,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one complete 500-step evaluation episode.
    """

    env, state = (
        create_environment(
            traffic_load=
                traffic_load,
            seed=
                seed,
        )
    )

    terminated = False
    truncated = False

    total_reward = 0.0

    action_counts = np.zeros(
        len(ACTIONS),
        dtype=np.int64,
    )

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # D3QN
        # ----------------------------------------------------

        if method == "D3QN":

            action = (
                select_d3qn_action(
                    state=
                        state,
                    network=
                        d3qn_network,
                    device=
                        device,
                )
            )

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------

        elif method == "PPO":

            predicted_action, _ = (
                ppo_model.predict(
                    state,
                    deterministic=True,
                )
            )

            action = int(
                np.asarray(
                    predicted_action
                ).item()
            )

        # ----------------------------------------------------
        # STATIC EQUAL ALLOCATION
        # ----------------------------------------------------

        elif method == "Baseline":

            action = (
                BASELINE_ACTION
            )

        else:

            raise ValueError(
                f"Unknown method: "
                f"{method}"
            )

        if not (
            0
            <= action
            < len(ACTIONS)
        ):
            raise RuntimeError(
                f"{method} selected "
                f"invalid action {action}."
            )

        action_counts[
            action
        ] += 1

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        total_reward += float(
            reward
        )

        state = next_state

    metrics = (
        env.get_episode_metrics()
    )

    env.close()

    return {
        "total_reward":
            total_reward,

        "metrics":
            metrics,

        "action_counts":
            action_counts,
    }


# ============================================================
# MAIN FINAL EVALUATION
# ============================================================

def main() -> None:

    os.makedirs(
        "results",
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 75)
    print("FINAL D3QN / PPO / BASELINE EVALUATION")
    print("=" * 75)

    print(
        f"Device          : "
        f"{device}"
    )

    print(
        f"D3QN model      : "
        f"{D3QN_MODEL_PATH}"
    )

    print(
        f"PPO model       : "
        f"{PPO_MODEL_PATH}"
    )

    print(
        f"Baseline action : "
        f"{BASELINE_ACTION}"
    )

    print(
        f"Seeds           : "
        f"{SEEDS[0]}-{SEEDS[-1]}"
    )

    print(
        f"Traffic loads   : "
        f"{TRAFFIC_LOADS}"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Confirm equal-allocation baseline
    # --------------------------------------------------------

    print(
        "Baseline allocation:",
        ACTIONS[
            BASELINE_ACTION
        ],
    )

    if ACTIONS[
        BASELINE_ACTION
    ] != {
        "eMBB": 25,
        "URLLC1": 25,
        "URLLC2": 25,
        "BE1": 25,
    }:
        raise RuntimeError(
            "Action 95 is no longer "
            "the equal-allocation action."
        )

    # --------------------------------------------------------
    # Load models once
    # --------------------------------------------------------

    d3qn_network = (
        load_d3qn(
            device
        )
    )

    if not os.path.exists(
        PPO_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Missing PPO model: "
            f"{PPO_MODEL_PATH}"
        )

    ppo_model = PPO.load(
        PPO_MODEL_PATH,
        device=device,
    )

    methods = [
        "D3QN",
        "PPO",
        "Baseline",
    ]

    detailed_rows = []
    run_summary_rows = []

    # ========================================================
    # LOW / MEDIUM / HIGH
    # ========================================================

    for traffic_load in (
        TRAFFIC_LOADS
    ):

        print()
        print("=" * 75)
        print(
            f"TRAFFIC LOAD: "
            f"{traffic_load.upper()}"
        )
        print("=" * 75)

        for method in methods:

            print()
            print(
                f"Evaluating "
                f"{method}..."
            )

            for run_number, seed in (
                enumerate(
                    SEEDS,
                    start=1,
                )
            ):

                result = (
                    run_episode(
                        method=
                            method,

                        traffic_load=
                            traffic_load,

                        seed=
                            seed,

                        d3qn_network=
                            d3qn_network,

                        ppo_model=
                            ppo_model,

                        device=
                            device,
                    )
                )

                total_reward = float(
                    result[
                        "total_reward"
                    ]
                )

                metrics = result[
                    "metrics"
                ]

                # --------------------------------------------
                # Per-slice results
                # --------------------------------------------

                slice_throughputs = []
                slice_latencies = []
                slice_jitters = []
                slice_plrs = []

                for slice_name in (
                    CONFIG[
                        "SLICES"
                    ]
                ):

                    slice_metrics = (
                        metrics[
                            slice_name
                        ]
                    )

                    throughput = float(
                        slice_metrics[
                            "average_throughput_percent"
                        ]
                    )

                    latency = float(
                        slice_metrics[
                            "average_latency_ms"
                        ]
                    )

                    jitter = float(
                        slice_metrics[
                            "jitter_ms"
                        ]
                    )

                    plr = float(
                        slice_metrics[
                            "plr_percent"
                        ]
                    )

                    slice_throughputs.append(
                        throughput
                    )

                    slice_latencies.append(
                        latency
                    )

                    slice_jitters.append(
                        jitter
                    )

                    slice_plrs.append(
                        plr
                    )

                    detailed_rows.append(
                        {
                            "traffic_load":
                                traffic_load,

                            "method":
                                method,

                            "run":
                                run_number,

                            "seed":
                                seed,

                            "slice":
                                slice_name,

                            "total_reward":
                                total_reward,

                            "throughput_percent":
                                throughput,

                            "latency_ms":
                                latency,

                            "jitter_ms":
                                jitter,

                            "plr_percent":
                                plr,

                            "queue_length":
                                int(
                                    slice_metrics[
                                        "queue_length"
                                    ]
                                ),

                            "total_throughput_bits":
                                float(
                                    slice_metrics[
                                        "total_throughput_bits"
                                    ]
                                ),
                        }
                    )

                # --------------------------------------------
                # System-level mean across four slices
                # --------------------------------------------

                mean_throughput = (
                    float(
                        np.mean(
                            slice_throughputs
                        )
                    )
                )

                mean_latency = (
                    float(
                        np.mean(
                            slice_latencies
                        )
                    )
                )

                mean_jitter = (
                    float(
                        np.mean(
                            slice_jitters
                        )
                    )
                )

                mean_plr = (
                    float(
                        np.mean(
                            slice_plrs
                        )
                    )
                )

                run_summary_rows.append(
                    {
                        "traffic_load":
                            traffic_load,

                        "method":
                            method,

                        "run":
                            run_number,

                        "seed":
                            seed,

                        "total_reward":
                            total_reward,

                        "mean_throughput_percent":
                            mean_throughput,

                        "mean_latency_ms":
                            mean_latency,

                        "mean_jitter_ms":
                            mean_jitter,

                        "mean_plr_percent":
                            mean_plr,
                    }
                )

                print(
                    f"  Run "
                    f"{run_number:2d}/"
                    f"{len(SEEDS)} | "
                    f"Seed={seed} | "
                    f"Reward="
                    f"{total_reward:.3f}"
                )

    # ========================================================
    # SAVE DETAILED CSV
    # ========================================================

    detailed_fields = [
        "traffic_load",
        "method",
        "run",
        "seed",
        "slice",
        "total_reward",
        "throughput_percent",
        "latency_ms",
        "jitter_ms",
        "plr_percent",
        "queue_length",
        "total_throughput_bits",
    ]

    with open(
        DETAILED_RESULT_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                detailed_fields,
        )

        writer.writeheader()

        writer.writerows(
            detailed_rows
        )

    # ========================================================
    # SAVE PER-RUN SUMMARY
    # ========================================================

    run_summary_fields = [
        "traffic_load",
        "method",
        "run",
        "seed",
        "total_reward",
        "mean_throughput_percent",
        "mean_latency_ms",
        "mean_jitter_ms",
        "mean_plr_percent",
    ]

    with open(
        RUN_SUMMARY_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                run_summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            run_summary_rows
        )

    # ========================================================
    # CREATE FINAL MEAN ± SD SUMMARY
    # ========================================================

    final_summary_rows = []

    for traffic_load in (
        TRAFFIC_LOADS
    ):

        for method in methods:

            selected = [
                row
                for row
                in run_summary_rows
                if (
                    row[
                        "traffic_load"
                    ]
                    == traffic_load
                    and
                    row[
                        "method"
                    ]
                    == method
                )
            ]

            rewards = np.asarray(
                [
                    row[
                        "total_reward"
                    ]
                    for row in selected
                ],
                dtype=float,
            )

            throughputs = np.asarray(
                [
                    row[
                        "mean_throughput_percent"
                    ]
                    for row in selected
                ],
                dtype=float,
            )

            latencies = np.asarray(
                [
                    row[
                        "mean_latency_ms"
                    ]
                    for row in selected
                ],
                dtype=float,
            )

            jitters = np.asarray(
                [
                    row[
                        "mean_jitter_ms"
                    ]
                    for row in selected
                ],
                dtype=float,
            )

            plrs = np.asarray(
                [
                    row[
                        "mean_plr_percent"
                    ]
                    for row in selected
                ],
                dtype=float,
            )

            final_summary_rows.append(
                {
                    "traffic_load":
                        traffic_load,

                    "method":
                        method,

                    "reward_mean":
                        float(
                            np.mean(
                                rewards
                            )
                        ),

                    "reward_std":
                        float(
                            np.std(
                                rewards
                            )
                        ),

                    "throughput_mean":
                        float(
                            np.mean(
                                throughputs
                            )
                        ),

                    "throughput_std":
                        float(
                            np.std(
                                throughputs
                            )
                        ),

                    "latency_mean":
                        float(
                            np.mean(
                                latencies
                            )
                        ),

                    "latency_std":
                        float(
                            np.std(
                                latencies
                            )
                        ),

                    "jitter_mean":
                        float(
                            np.mean(
                                jitters
                            )
                        ),

                    "jitter_std":
                        float(
                            np.std(
                                jitters
                            )
                        ),

                    "plr_mean":
                        float(
                            np.mean(
                                plrs
                            )
                        ),

                    "plr_std":
                        float(
                            np.std(
                                plrs
                            )
                        ),
                }
            )

    final_fields = [
        "traffic_load",
        "method",
        "reward_mean",
        "reward_std",
        "throughput_mean",
        "throughput_std",
        "latency_mean",
        "latency_std",
        "jitter_mean",
        "jitter_std",
        "plr_mean",
        "plr_std",
    ]

    with open(
        FINAL_SUMMARY_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                final_fields,
        )

        writer.writeheader()

        writer.writerows(
            final_summary_rows
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    for row in final_summary_rows:

        print(
            f"{row['traffic_load']:6s} | "
            f"{row['method']:8s} | "
            f"Reward="
            f"{row['reward_mean']:.2f} | "
            f"Throughput="
            f"{row['throughput_mean']:.2f}% | "
            f"Latency="
            f"{row['latency_mean']:.2f} ms | "
            f"Jitter="
            f"{row['jitter_mean']:.2f} ms | "
            f"PLR="
            f"{row['plr_mean']:.2f}%"
        )

    print()
    print("=" * 90)
    print("FINAL EVALUATION COMPLETED")
    print("=" * 90)

    print(
        "Detailed results :",
        DETAILED_RESULT_PATH,
    )

    print(
        "Run summaries    :",
        RUN_SUMMARY_PATH,
    )

    print(
        "Final summary    :",
        FINAL_SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()