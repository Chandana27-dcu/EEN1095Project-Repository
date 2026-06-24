from src.environment import SlicingEnv

for action_index in range(5):
    env = SlicingEnv()
    env.reset()
    total_reward = 0

    for t in range(env.max_time):
        state, reward, done, _ = env.step(action_index)
        total_reward += reward

    print("\nAction:", action_index)
    print("Allocation:", env.actions[action_index])
    print("Total Reward:", total_reward)
    print("Final State:", state)
    print("Throughput:", env.metrics["throughput"])
    print("PLR:", env.metrics["plr"])
    print("------------------------")