import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment import SlicingEnv
from src.config import CONFIG


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


def train_d3qn():
    env = SlicingEnv()

    state_dim = len(env.get_state())
    action_dim = len(env.actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG["LR"])
    replay_buffer = ReplayBuffer(CONFIG["BUFFER_SIZE"])

    epsilon = CONFIG["EPS_START"]
    episode_rewards = []

    for episode in range(CONFIG["EPISODES"]):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)

        total_reward = 0

        for t in range(env.max_time):
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            replay_buffer.push(
                state.cpu().numpy(),
                action,
                reward,
                next_state_tensor.cpu().numpy(),
                done
            )

            state = next_state_tensor
            total_reward += reward

            if len(replay_buffer) >= CONFIG["BATCH_SIZE"]:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    CONFIG["BATCH_SIZE"]
                )

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q = policy_net(states).gather(
                    1, actions.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1)
                    next_q = target_net(next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)

                    target_q = rewards + CONFIG["GAMMA"] * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(CONFIG["EPS_END"], epsilon * CONFIG["EPS_DECAY"])
        episode_rewards.append(total_reward)

        if episode % CONFIG["TARGET_UPDATE"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(
            f"Episode {episode}, "
            f"Reward {total_reward:.2f}, "
            f"Epsilon {epsilon:.3f}"
        )

    print("Reached save section")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(policy_net.state_dict(), "models/dqn_model.pth")
    np.save("results/rewards.npy", np.array(episode_rewards))

    print("Model saved as models/dqn_model.pth")
    print("Rewards saved as results/rewards.npy")


if __name__ == "__main__":
    train_d3qn()