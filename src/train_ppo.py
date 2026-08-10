from __future__ import annotations

import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.config_ppo import CONFIG
from src.environment_ppo import NetworkSlicingPPOEnv


# ============================================================
# OUTPUT PATHS
# ============================================================

LOAD_SCENARIO = CONFIG["LOAD_SCENARIO"]

MODEL_PATH = CONFIG["MODEL_PATH"]

TRAINING_HISTORY_PATH = (
    f"results/ppo_{LOAD_SCENARIO}_training_history.csv"
)


# ============================================================
# CALLBACK TO RECORD EPISODE REWARDS
# ============================================================

class RewardHistoryCallback(BaseCallback):
    """
    Record PPO episode rewards during training.

    The saved CSV will later be used for the
    D3QN vs PPO convergence plot.
    """

    def __init__(
        self,
        output_path: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.output_path = output_path

        self.episode_number = 0

        self.history: list[
            dict[str, float | int]
        ] = []

    def _on_step(self) -> bool:

        infos = self.locals.get(
            "infos",
            [],
        )

        for info in infos:

            if "episode" in info:

                self.episode_number += 1

                episode_reward = float(
                    info["episode"]["r"]
                )

                episode_length = int(
                    info["episode"]["l"]
                )

                self.history.append(
                    {
                        "episode":
                            self.episode_number,

                        "reward":
                            episode_reward,

                        "episode_length":
                            episode_length,

                        "timesteps":
                            self.num_timesteps,
                    }
                )

        return True

    def _on_training_end(self) -> None:

        if not self.history:
            print(
                "Warning: no PPO episode rewards "
                "were recorded."
            )
            return

        os.makedirs(
            os.path.dirname(
                self.output_path
            ),
            exist_ok=True,
        )

        with open(
            self.output_path,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            field_names = [
                "episode",
                "reward",
                "episode_length",
                "timesteps",
            ]

            writer = csv.DictWriter(
                csv_file,
                fieldnames=field_names,
            )

            writer.writeheader()

            writer.writerows(
                self.history
            )

        print(
            f"PPO training history saved to: "
            f"{self.output_path}"
        )


# ============================================================
# PPO TRAINING
# ============================================================

def train_ppo() -> None:

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True,
    )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = NetworkSlicingPPOEnv()

    # Validate Gymnasium compatibility.
    check_env(
        env,
        warn=True,
    )

    # Monitor is required so Stable-Baselines3
    # records episode reward and episode length.
    env = Monitor(
        env
    )

    # --------------------------------------------------------
    # PPO configuration
    # --------------------------------------------------------

    ppo_config = CONFIG[
        "PPO"
    ]

    print("=" * 70)
    print("PPO TRAINING")
    print("=" * 70)

    print(
        f"Traffic load        : "
        f"{LOAD_SCENARIO}"
    )

    print(
        f"State size          : "
        f"{CONFIG['STATE_SIZE']}"
    )

    print(
        f"Action count        : "
        f"{CONFIG['NUMBER_OF_ACTIONS']}"
    )

    print(
        f"Total RBs           : "
        f"{CONFIG['TOTAL_RB']}"
    )

    print(
        f"Training timesteps  : "
        f"{ppo_config['total_timesteps']}"
    )

    print(
        f"Model path          : "
        f"{MODEL_PATH}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # PPO model
    #
    # Because the environment action space is Discrete(155),
    # Stable-Baselines3 automatically uses a categorical policy.
    # --------------------------------------------------------

    model = PPO(
        policy="MlpPolicy",

        env=env,

        learning_rate=float(
            ppo_config[
                "learning_rate"
            ]
        ),

        n_steps=int(
            ppo_config[
                "n_steps"
            ]
        ),

        batch_size=int(
            ppo_config[
                "batch_size"
            ]
        ),

        gamma=float(
            ppo_config[
                "gamma"
            ]
        ),

        gae_lambda=float(
            ppo_config[
                "gae_lambda"
            ]
        ),

        clip_range=float(
            ppo_config[
                "clip_range"
            ]
        ),

        ent_coef=float(
            ppo_config[
                "ent_coef"
            ]
        ),

        vf_coef=float(
            ppo_config[
                "vf_coef"
            ]
        ),

        max_grad_norm=float(
            ppo_config[
                "max_grad_norm"
            ]
        ),

        n_epochs=int(
            ppo_config[
                "n_epochs"
            ]
        ),

        seed=int(
            CONFIG["SEED"]
        ),

        verbose=1,

        policy_kwargs={
            "net_arch": [
                int(
                    ppo_config[
                        "hidden_neurons"
                    ]
                ),
                int(
                    ppo_config[
                        "hidden_neurons"
                    ]
                ),
            ]
        },
    )

    # --------------------------------------------------------
    # Reward-history callback
    # --------------------------------------------------------

    reward_callback = (
        RewardHistoryCallback(
            output_path=
                TRAINING_HISTORY_PATH
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.learn(
        total_timesteps=int(
            ppo_config[
                "total_timesteps"
            ]
        ),

        callback=
            reward_callback,

        progress_bar=False,
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    env.close()

    print()
    print("=" * 70)
    print("PPO TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Model saved to    : "
        f"{MODEL_PATH}"
    )

    print(
        f"Training history  : "
        f"{TRAINING_HISTORY_PATH}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_ppo()