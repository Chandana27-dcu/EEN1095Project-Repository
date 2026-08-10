from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def moving_average(values, window):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


rewards_file = RESULTS_DIR / "d3q_medium_training_rewards.npy"
losses_file = RESULTS_DIR / "d3q_medium_training_losses.npy"

print("Loading:", rewards_file)
rewards = np.load(rewards_file)

episodes = np.arange(1, len(rewards) + 1)
reward_average = moving_average(rewards, 10)

plt.figure(figsize=(9, 5))
plt.plot(episodes, rewards, alpha=0.35, label="Episode reward")

if len(rewards) >= 10:
    plt.plot(
        np.arange(10, len(rewards) + 1),
        reward_average,
        linewidth=2,
        label="10-episode moving average",
    )

plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("D3QN Training Reward - Medium Traffic Load")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "d3q_medium_training_reward.png",
    dpi=300,
)
plt.close()

print("Loading:", losses_file)
losses = np.load(losses_file)

updates = np.arange(1, len(losses) + 1)
loss_window = min(500, max(1, len(losses)))
loss_average = moving_average(losses, loss_window)

plt.figure(figsize=(9, 5))
plt.plot(updates, losses, alpha=0.2, label="Training loss")

if len(losses) >= loss_window:
    plt.plot(
        np.arange(loss_window, len(losses) + 1),
        loss_average,
        linewidth=2,
        label=f"{loss_window}-update moving average",
    )

plt.xlabel("Training update")
plt.ylabel("Loss")
plt.title("D3QN Training Loss - Medium Traffic Load")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "d3q_medium_training_loss.png",
    dpi=300,
)
plt.close()

print("Graphs created successfully.")
print("figures/d3q_medium_training_reward.png")
print("figures/d3q_medium_training_loss.png")
