from src.environment_d3q import NetworkSlicingD3QEnv


NUM_ACTIONS_TO_TEST = 5


def run_baseline(action_index: int) -> None:
    env = NetworkSlicingD3QEnv()

    observation, info = env.reset()

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(
            action_index
        )
        total_reward += float(reward)

    results = env.get_episode_metrics()

    print("\n" + "=" * 70)
    print(f"Baseline Action Index: {action_index}")
    print(f"Total Reward: {total_reward:.4f}")
    print("=" * 70)

    for slice_name, metrics in results.items():
        print(f"\nSlice: {slice_name}")

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")


def main() -> None:
    for action_index in range(NUM_ACTIONS_TO_TEST):
        run_baseline(action_index)


if __name__ == "__main__":
    main()