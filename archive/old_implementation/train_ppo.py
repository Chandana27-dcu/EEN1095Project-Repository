import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.config_ppo import CONFIG
from src.environment_ppo import NetworkSlicingPPOEnv


def train_ppo():
    os.makedirs("models", exist_ok=True)

    env = NetworkSlicingPPOEnv()
    check_env(env)

    ppo_config = CONFIG["PPO"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        gamma=ppo_config["gamma"],
        verbose=1,
        policy_kwargs={
            "net_arch": [
                ppo_config["hidden_neurons"],
                ppo_config["hidden_neurons"],
            ]
        },
    )

    model.learn(
        total_timesteps=ppo_config["total_timesteps"]
    )

    model.save("models/ppo_network_slicing")

    print("PPO model saved successfully.")


if __name__ == "__main__":
    train_ppo()
