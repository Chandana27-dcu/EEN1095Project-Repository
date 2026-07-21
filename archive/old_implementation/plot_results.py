import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# 1. Episode vs Reward
rewards = np.load("results/rewards.npy")

plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Reward")
plt.grid(True)
plt.savefig("plots/episode_vs_reward.png")
plt.close()

# 2. Baseline vs DQN
methods = ["Best Baseline", "DQN"]
values = [84346, 70897]

plt.figure()
plt.bar(methods, values)
plt.ylabel("Total Reward")
plt.title("Baseline vs DQN Reward")
plt.grid(axis="y")
plt.savefig("plots/baseline_vs_dqn.png")
plt.close()

# 3. Action selection count
actions = ["A0", "A1", "A2", "A3", "A4"]
counts = [59, 0, 16, 2, 123]

plt.figure()
plt.bar(actions, counts)
plt.xlabel("Action")
plt.ylabel("Selected Count")
plt.title("DQN Action Selection Count")
plt.grid(axis="y")
plt.savefig("plots/action_count.png")
plt.close()

print("Graphs saved in plots folder.")