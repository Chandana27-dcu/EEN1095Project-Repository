import torch
import numpy as np

from src.environment import SlicingEnv
from src.train import DuelingDQN


def evaluate_dqn():
    env = SlicingEnv()

    state_dim = len(env.get_state())
    action_dim = len(env.actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DuelingDQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load("models/dqn_model.pth", map_location=device))
    model.eval()

    state = env.reset()
    total_reward = 0
    action_count = {i: 0 for i in range(action_dim)}

    for t in range(env.max_time):
        state_tensor = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            q_values = model(state_tensor.unsqueeze(0))
            action = q_values.argmax().item()

        action_count[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    print("===== DQN Evaluation Result =====")
    print("DQN Evaluation Reward:", total_reward)
    print("Action Count:", action_count)
    print("Final State:", state)
    print("Throughput:", env.metrics["throughput"])
    print("PLR:", env.metrics["plr"])

    print("\nSelected RB allocations:")
    for action_id, count in action_count.items():
        print(action_id, env.actions[action_id], "used", count, "times")


if __name__ == "__main__":
    evaluate_dqn()