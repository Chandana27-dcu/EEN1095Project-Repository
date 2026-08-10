from __future__ import annotations

import csv
import os

from src.baseline_fixed import StaticEqualAllocationBaseline
from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv


LOAD_SCENARIO = CONFIG["LOAD_SCENARIO"]

RESULT_PATH = (
    f"results/baseline_{LOAD_SCENARIO}_results.csv"
)


def evaluate_baseline() -> None:
    """
    Evaluate the Static QoS-Aware Allocation baseline
    using the same environment as D3QN.
    """

    os.makedirs(
        "results",
        exist_ok=True,
    )

    baseline = StaticEqualAllocationBaseline()

    number_of_runs = int(
        CONFIG["EVALUATION"]["number_of_runs"]
    )

    base_seed = int(
        CONFIG["EVALUATION"]["base_seed"]
    )

    all_results = []

    print("=" * 70)
    print("STATIC QoS-AWARE ALLOCATION BASELINE EVALUATION")
    print("=" * 70)

    print(
        f"Traffic load : {LOAD_SCENARIO}"
    )

    print(
        f"Runs         : {number_of_runs}"
    )

    print(
        f"Action       : {baseline.action_index}"
    )

    print("=" * 70)

    # ========================================================
    # MULTIPLE EVALUATION RUNS
    # ========================================================

    for run in range(
        number_of_runs
    ):

        seed = (
            base_seed + run
        )

        env = (
            NetworkSlicingD3QEnv()
        )

        state, _ = env.reset(
            seed=seed
        )

        terminated = False
        truncated = False

        total_reward = 0.0

        # ----------------------------------------------------
        # Episode
        # ----------------------------------------------------

        while not (
            terminated
            or truncated
        ):

            action = baseline.select_action(
                state
            )

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

        print()
        print(
            f"Run {run + 1}/{number_of_runs}"
        )

        print(
            f"Seed         : {seed}"
        )

        print(
            f"Action       : "
            f"{baseline.action_index}"
        )

        print(
            f"Total reward : "
            f"{total_reward:.4f}"
        )

        # ----------------------------------------------------
        # Per-slice metrics
        # ----------------------------------------------------

        for slice_name in CONFIG[
            "SLICES"
        ]:

            slice_metrics = (
                metrics[
                    slice_name
                ]
            )

            print(
                f"{slice_name:7s} | "
                f"Throughput="
                f"{slice_metrics['average_throughput_percent']:.2f}% | "
                f"Latency="
                f"{slice_metrics['average_latency_ms']:.2f} ms | "
                f"Jitter="
                f"{slice_metrics['jitter_ms']:.2f} ms | "
                f"PLR="
                f"{slice_metrics['plr_percent']:.2f}%"
            )

            all_results.append(
                {
                    "run":
                        run + 1,

                    "seed":
                        seed,

                    "traffic_load":
                        LOAD_SCENARIO,

                    "method":
                        baseline.name,

                    "slice":
                        slice_name,

                    "action_index":
                        baseline.action_index,

                    "total_reward":
                        total_reward,

                    "throughput_percent":
                        slice_metrics[
                            "average_throughput_percent"
                        ],

                    "latency_ms":
                        slice_metrics[
                            "average_latency_ms"
                        ],

                    "jitter_ms":
                        slice_metrics[
                            "jitter_ms"
                        ],

                    "plr_percent":
                        slice_metrics[
                            "plr_percent"
                        ],

                    "queue_length":
                        slice_metrics[
                            "queue_length"
                        ],

                    "total_throughput_bits":
                        slice_metrics[
                            "total_throughput_bits"
                        ],

                    "final_allocation_percent":
                        slice_metrics[
                            "final_allocation_percent"
                        ],
                }
            )

        env.close()

    # ========================================================
    # SAVE CSV
    # ========================================================

    field_names = [
        "run",
        "seed",
        "traffic_load",
        "method",
        "slice",
        "action_index",
        "total_reward",
        "throughput_percent",
        "latency_ms",
        "jitter_ms",
        "plr_percent",
        "queue_length",
        "total_throughput_bits",
        "final_allocation_percent",
    ]

    with open(
        RESULT_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print()
    print("=" * 70)
    print("BASELINE EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{RESULT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate_baseline()