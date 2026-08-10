from __future__ import annotations

import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src.config_d3q import CONFIG
from src.environment_d3q import NetworkSlicingD3QEnv
from src.network_d3q import DuelingQNetwork
from src.prioritized_replay import PrioritizedReplayBuffer


MODEL_PATH = "models/d3q_network_slicing.pth"
REWARD_PATH = "results/d3q_training_rewards.npy"
LOSS_PATH = "results/d3q_training_losses.npy"


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_beta(training_step: int) -> float:
    """Increase PER beta gradually from its initial value to 1.0."""
    beta_start = float(CONFIG["PER_BETA_START"])
    beta_frames = int(CONFIG["PER_BETA_FRAMES"])

    progress = min(
        training_step / max(beta_frames, 1),
        1.0,
    )

    return float(
        beta_start
        + progress * (1.0 - beta_start)
    )


def select_action(
    state: np.ndarray,
    network: DuelingQNetwork,
    epsilon: float,
    action_size: int,
    device: torch.device,
) -> int:
    """Select an action using epsilon-greedy exploration."""
    if random.random() < epsilon:
        return random.randrange(action_size)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(state_tensor)

    return int(
        torch.argmax(q_values, dim=1).item()
    )


def optimize_network(
    online_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    replay_buffer: PrioritizedReplayBuffer,
    optimizer: Adam,
    device: torch.device,
    training_step: int,
) -> float:
    """Run one Double-DQN update using prioritized replay."""
    batch_size = int(CONFIG["BATCH_SIZE"])
    gamma = float(CONFIG["GAMMA"])
    max_grad_norm = float(CONFIG["MAX_GRAD_NORM"])

    beta = calculate_beta(training_step)

    batch = replay_buffer.sample(
        batch_size=batch_size,
        beta=beta,
    )

    states = torch.as_tensor(
        batch["states"],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        batch["actions"],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        batch["rewards"],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.as_tensor(
        batch["next_states"],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor(
        batch["dones"],
        dtype=torch.float32,
        device=device,
    )
    importance_weights = torch.as_tensor(
        batch["weights"],
        dtype=torch.float32,
        device=device,
    )

    current_q_values = (
        online_network(states)
        .gather(1, actions)
        .squeeze(1)
    )

    with torch.no_grad():
        # Online network selects the best next action.
        next_action_indices = (
            online_network(next_states)
            .argmax(dim=1, keepdim=True)
        )

        # Target network evaluates that selected action.
        next_q_values = (
            target_network(next_states)
            .gather(1, next_action_indices)
            .squeeze(1)
        )

        target_q_values = (
            rewards
            + gamma
            * next_q_values
            * (1.0 - dones)
        )

    td_errors = target_q_values - current_q_values

    element_losses = nn.functional.smooth_l1_loss(
        current_q_values,
        target_q_values,
        reduction="none",
    )

    loss = torch.mean(
        importance_weights * element_losses
    )

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_grad_norm,
    )

    optimizer.step()

    replay_buffer.update_priorities(
        indices=batch["indices"],
        td_errors=(
            td_errors.detach().cpu().numpy()
        ),
    )

    return float(loss.item())


def train_d3q() -> None:
    seed = 42
    set_random_seeds(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    env = NetworkSlicingD3QEnv()

    state_size = int(CONFIG["STATE_SIZE"])
    action_size = int(CONFIG["NUMBER_OF_ACTIONS"])
    hidden_size = int(CONFIG["HIDDEN_SIZE"])

    online_network = DuelingQNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
    ).to(device)

    target_network = DuelingQNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
    ).to(device)

    target_network.load_state_dict(
        online_network.state_dict()
    )
    target_network.eval()

    optimizer = Adam(
        online_network.parameters(),
        lr=float(CONFIG["LEARNING_RATE"]),
    )

    replay_buffer = PrioritizedReplayBuffer(
        capacity=int(CONFIG["REPLAY_BUFFER_SIZE"]),
        alpha=float(CONFIG["PER_ALPHA"]),
        epsilon=float(CONFIG["PER_EPSILON"]),
    )

    episodes = int(CONFIG["EPISODES"])
    epsilon = float(CONFIG["EPS_START"])
    epsilon_end = float(CONFIG["EPS_END"])
    epsilon_decay = float(CONFIG["EPS_DECAY"])

    target_update = int(CONFIG["TARGET_UPDATE"])
    learning_starts = int(CONFIG["LEARNING_STARTS"])
    train_frequency = int(CONFIG["TRAIN_FREQUENCY"])

    training_step = 0
    episode_rewards: list[float] = []
    training_losses: list[float] = []
    recent_rewards: deque[float] = deque(maxlen=50)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)

        episode_reward = 0.0
        episode_losses: list[float] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = select_action(
                state=state,
                network=online_network,
                epsilon=epsilon,
                action_size=action_size,
                device=device,
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            done = terminated or truncated

            replay_buffer.add(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=done,
            )

            state = next_state
            episode_reward += float(reward)
            training_step += 1

            minimum_buffer_size = max(
                learning_starts,
                int(CONFIG["BATCH_SIZE"]),
            )

            if (
                len(replay_buffer) >= minimum_buffer_size
                and training_step % train_frequency == 0
            ):
                loss = optimize_network(
                    online_network=online_network,
                    target_network=target_network,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    device=device,
                    training_step=training_step,
                )

                episode_losses.append(loss)
                training_losses.append(loss)

        epsilon = max(
            epsilon_end,
            epsilon * epsilon_decay,
        )

        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        if episode % target_update == 0:
            target_network.load_state_dict(
                online_network.state_dict()
            )

        average_recent_reward = float(
            np.mean(recent_rewards)
        )

        average_episode_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        print(
            f"Episode {episode:4d}/{episodes} | "
            f"Reward={episode_reward:8.3f} | "
            f"Average50={average_recent_reward:8.3f} | "
            f"Loss={average_episode_loss:8.5f} | "
            f"Epsilon={epsilon:.4f} | "
            f"Buffer={len(replay_buffer)}"
        )

        if episode % 100 == 0:
            checkpoint_path = (
                f"models/d3q_checkpoint_{episode}.pth"
            )

            torch.save(
                {
                    "episode": episode,
                    "model_state_dict":
                        online_network.state_dict(),
                    "target_state_dict":
                        target_network.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "epsilon": epsilon,
                    "training_step": training_step,
                    "config": CONFIG,
                },
                checkpoint_path,
            )

    torch.save(
        {
            "model_state_dict":
                online_network.state_dict(),
            "target_state_dict":
                target_network.state_dict(),
            "state_size": state_size,
            "action_size": action_size,
            "hidden_size": hidden_size,
            "epsilon": epsilon,
            "training_step": training_step,
            "config": CONFIG,
        },
        MODEL_PATH,
    )

    np.save(
        REWARD_PATH,
        np.asarray(
            episode_rewards,
            dtype=np.float32,
        ),
    )

    np.save(
        LOSS_PATH,
        np.asarray(
            training_losses,
            dtype=np.float32,
        ),
    )

    env.close()

    print("\nD3QN training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Rewards saved to: {REWARD_PATH}")
    print(f"Losses saved to: {LOSS_PATH}")


if __name__ == "__main__":
    train_d3q()