from stable_baselines3 import PPO

from src.environment_ppo import NetworkSlicingPPOEnv


MODEL_PATH = "models/ppo_network_slicing"


def evaluate_ppo():
    env = NetworkSlicingPPOEnv()
    model = PPO.load(MODEL_PATH)

    observation, _ = env.reset(seed=42)

    total_reward = 0.0
    final_info = {}

    for _ in range(env.max_time):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)
        final_info = info

        if terminated or truncated:
            break

    episode_metrics = env.get_episode_metrics()

    print("===== PPO Evaluation Result =====")
    print(f"Total Reward: {total_reward:.4f}")

    average_reward = total_reward / env.max_time
    print(f"Average Reward per Step: {average_reward:.4f}")

    print("\nFinal Allocation Percentage:")

    allocation_percent = final_info.get(
        "allocation_percent",
        {}
    )

    allocation_total = 0.0

    for slice_name in env.slices:
        allocation = float(
            allocation_percent.get(slice_name, 0.0)
        )

        allocation_total += allocation

        print(
            f"{slice_name}: {allocation:.2f}%"
        )

    print(
        f"Total Allocation: {allocation_total:.2f}%"
    )

    print("\nFinal Metrics for All 4 Slices:")

    for slice_name in env.slices:
        metrics = episode_metrics[slice_name]

        print(f"\nSlice: {slice_name}")

        print(
            f"Average Throughput (%): "
            f"{metrics['average_throughput_percent']:.2f}"
        )

        print(
            f"Current Throughput (%): "
            f"{metrics['current_throughput_percent']:.2f}"
        )

        print(
            f"Average Latency (ms): "
            f"{metrics['average_latency_ms']:.4f}"
        )

        print(
            f"Jitter (ms): "
            f"{metrics['jitter_ms']:.4f}"
        )

        print(
            f"PLR (%): "
            f"{metrics['plr_percent']:.4f}"
        )

        print(
            f"Queue Length: "
            f"{metrics['queue_length']}"
        )

        print(
            f"Channel Condition: "
            f"{metrics['channel_condition']:.4f}"
        )

        print(
            f"Arrivals: "
            f"{metrics['arrivals']}"
        )

        print(
            f"Dropped Packets: "
            f"{metrics['dropped']}"
        )

        print(
            f"Total Throughput Bits: "
            f"{metrics['total_throughput_bits']:.0f}"
        )


if __name__ == "__main__":
    evaluate_ppo()
