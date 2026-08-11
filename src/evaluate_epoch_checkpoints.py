from __future__ import annotations

import csv
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv
from src.network_d3q import DuelingQNetwork


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_DIR = "models/epoch_checkpoints"
RESULT_DIR = "results/epoch_analysis"

TRAFFIC_LOAD = "medium"

CHECKPOINT_EPISODES = list(
    range(10, 201, 10)
)

TEST_SEEDS = list(
    range(42, 52)
)

BASELINE_ACTION = 95


DETAILED_PATH = os.path.join(
    RESULT_DIR,
    "checkpoint_testing_detailed.csv",
)

SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    "checkpoint_testing_summary.csv",
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_environment(
    seed: int,
) -> tuple[
    NetworkSlicingD3QEnv,
    np.ndarray,
]:

    env = NetworkSlicingD3QEnv()

    # Force Medium traffic for checkpoint evaluation.
    env.traffic_load = TRAFFIC_LOAD

    env.traffic_config = (
        CONFIG[
            "TRAFFIC_SCENARIOS"
        ][
            TRAFFIC_LOAD
        ].copy()
    )

    state, _ = env.reset(
        seed=seed
    )

    return env, state


# ============================================================
# LOAD D3QN CHECKPOINT
# ============================================================

def load_d3qn_checkpoint(
    episode: int,
    device: torch.device,
) -> DuelingQNetwork:

    path = os.path.join(
        CHECKPOINT_DIR,
        f"d3qn_ep_{episode:03d}.pth",
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing D3QN checkpoint: {path}"
        )

    checkpoint = torch.load(
        path,
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
            CONFIG[
                "D3QN"
            ][
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
# LOAD PPO CHECKPOINT
# ============================================================

def load_ppo_checkpoint(
    episode: int,
    device: torch.device,
) -> PPO:

    path = os.path.join(
        CHECKPOINT_DIR,
        f"ppo_ep_{episode:03d}.zip",
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing PPO checkpoint: {path}"
        )

    model = PPO.load(
        path,
        device=device,
    )

    return model


# ============================================================
# D3QN GREEDY ACTION
# ============================================================

def select_d3qn_action(
    state: np.ndarray,
    network: DuelingQNetwork,
    device: torch.device,
) -> int:

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = network(
            state_tensor
        )

    action = int(
        torch.argmax(
            q_values,
            dim=1,
        ).item()
    )

    return action


# ============================================================
# RUN ONE TEST EPISODE
# ============================================================

def run_test_episode(
    method: str,
    seed: int,
    device: torch.device,
    d3qn_network: DuelingQNetwork | None = None,
    ppo_model: PPO | None = None,
) -> dict:

    env, state = create_environment(
        seed=seed
    )

    terminated = False
    truncated = False

    total_reward = 0.0

    while not (
        terminated
        or truncated
    ):

        # ----------------------------------------------------
        # D3QN
        # ----------------------------------------------------

        if method == "D3QN":

            if d3qn_network is None:
                raise ValueError(
                    "D3QN network is missing."
                )

            action = select_d3qn_action(
                state=state,
                network=d3qn_network,
                device=device,
            )

        # ----------------------------------------------------
        # PPO
        # ----------------------------------------------------

        elif method == "PPO":

            if ppo_model is None:
                raise ValueError(
                    "PPO model is missing."
                )

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

        elif method == "SEA":

            action = BASELINE_ACTION

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        (
            next_state,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(
            action
        )

        total_reward += float(
            reward
        )

        state = next_state

    # --------------------------------------------------------
    # Get final episode metrics
    # --------------------------------------------------------

    metrics = (
        env.get_episode_metrics()
    )

    throughputs = []
    latencies = []
    jitters = []
    plrs = []

    for slice_name in CONFIG[
        "SLICES"
    ]:

        slice_metrics = metrics[
            slice_name
        ]

        throughputs.append(
            float(
                slice_metrics[
                    "average_throughput_percent"
                ]
            )
        )

        latencies.append(
            float(
                slice_metrics[
                    "average_latency_ms"
                ]
            )
        )

        jitters.append(
            float(
                slice_metrics[
                    "jitter_ms"
                ]
            )
        )

        plrs.append(
            float(
                slice_metrics[
                    "plr_percent"
                ]
            )
        )

    result = {

        "reward":
            float(
                total_reward
            ),

        "throughput":
            float(
                np.mean(
                    throughputs
                )
            ),

        "latency":
            float(
                np.mean(
                    latencies
                )
            ),

        "jitter":
            float(
                np.mean(
                    jitters
                )
            ),

        "plr":
            float(
                np.mean(
                    plrs
                )
            ),
    }

    env.close()

    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path: str,
    rows: list[dict],
    fieldnames: list[str],
) -> None:

    with open(
        path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 75)
    print("CHECKPOINT TESTING")
    print("=" * 75)

    print(
        f"Traffic load : {TRAFFIC_LOAD}"
    )

    print(
        f"Test seeds   : "
        f"{TEST_SEEDS[0]}-"
        f"{TEST_SEEDS[-1]}"
    )

    print(
        f"Checkpoints  : "
        f"{len(CHECKPOINT_EPISODES)}"
    )

    print(
        f"Device       : {device}"
    )

    print("=" * 75)

    detailed_rows = []
    summary_rows = []

    # ========================================================
    # EVALUATE EVERY CHECKPOINT
    # ========================================================

    for episode in CHECKPOINT_EPISODES:

        print()
        print("=" * 75)

        print(
            f"TRAINING EPISODE "
            f"{episode}"
        )

        print("=" * 75)

        # ----------------------------------------------------
        # Load learning models
        # ----------------------------------------------------

        d3qn_network = (
            load_d3qn_checkpoint(
                episode=episode,
                device=device,
            )
        )

        ppo_model = (
            load_ppo_checkpoint(
                episode=episode,
                device=device,
            )
        )

        # ----------------------------------------------------
        # Three methods
        # ----------------------------------------------------

        for method in [
            "D3QN",
            "PPO",
            "SEA",
        ]:

            method_results = []

            print(
                f"Testing {method}..."
            )

            # ------------------------------------------------
            # Same 10 seeds for every method/checkpoint
            # ------------------------------------------------

            for run_number, seed in enumerate(
                TEST_SEEDS,
                start=1,
            ):

                result = run_test_episode(

                    method=method,

                    seed=seed,

                    device=device,

                    d3qn_network=(
                        d3qn_network
                        if method == "D3QN"
                        else None
                    ),

                    ppo_model=(
                        ppo_model
                        if method == "PPO"
                        else None
                    ),
                )

                method_results.append(
                    result
                )

                detailed_rows.append(
                    {
                        "episode":
                            episode,

                        "method":
                            method,

                        "run":
                            run_number,

                        "seed":
                            seed,

                        "reward":
                            result[
                                "reward"
                            ],

                        "throughput_percent":
                            result[
                                "throughput"
                            ],

                        "latency_ms":
                            result[
                                "latency"
                            ],

                        "jitter_ms":
                            result[
                                "jitter"
                            ],

                        "plr_percent":
                            result[
                                "plr"
                            ],
                    }
                )

            # ------------------------------------------------
            # Calculate mean ± standard deviation
            # ------------------------------------------------

            rewards = np.asarray(
                [
                    row["reward"]
                    for row
                    in method_results
                ],
                dtype=np.float64,
            )

            throughputs = np.asarray(
                [
                    row["throughput"]
                    for row
                    in method_results
                ],
                dtype=np.float64,
            )

            latencies = np.asarray(
                [
                    row["latency"]
                    for row
                    in method_results
                ],
                dtype=np.float64,
            )

            jitters = np.asarray(
                [
                    row["jitter"]
                    for row
                    in method_results
                ],
                dtype=np.float64,
            )

            plrs = np.asarray(
                [
                    row["plr"]
                    for row
                    in method_results
                ],
                dtype=np.float64,
            )

            summary_row = {

                "episode":
                    episode,

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

            summary_rows.append(
                summary_row
            )

            print(
                f"{method:5s} | "
                f"Reward="
                f"{summary_row['reward_mean']:.3f} | "
                f"Throughput="
                f"{summary_row['throughput_mean']:.2f}% | "
                f"Latency="
                f"{summary_row['latency_mean']:.2f} ms | "
                f"Jitter="
                f"{summary_row['jitter_mean']:.2f} ms | "
                f"PLR="
                f"{summary_row['plr_mean']:.2f}%"
            )

    # ========================================================
    # SAVE DETAILED DATA
    # ========================================================

    save_csv(

        path=
            DETAILED_PATH,

        rows=
            detailed_rows,

        fieldnames=[
            "episode",
            "method",
            "run",
            "seed",
            "reward",
            "throughput_percent",
            "latency_ms",
            "jitter_ms",
            "plr_percent",
        ],
    )

    # ========================================================
    # SAVE SUMMARY DATA
    # ========================================================

    save_csv(

        path=
            SUMMARY_PATH,

        rows=
            summary_rows,

        fieldnames=[
            "episode",
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
        ],
    )

    print()
    print("=" * 75)

    print(
        "CHECKPOINT TESTING COMPLETED"
    )

    print("=" * 75)

    print(
        f"Detailed results : "
        f"{DETAILED_PATH}"
    )

    print(
        f"Summary results  : "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()