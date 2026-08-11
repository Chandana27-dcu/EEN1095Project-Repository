from __future__ import annotations

import argparse
import csv
import os
from collections import deque

import numpy as np
import torch
from torch.optim import Adam

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.config_d3q import CONFIG as D3QN_CONFIG
from src.config_ppo import CONFIG as PPO_CONFIG

from src.environment_d3q import NetworkSlicingD3QEnv
from src.environment_ppo import NetworkSlicingPPOEnv

from src.network_d3q import DuelingQNetwork
from src.prioritized_replay import PrioritizedReplayBuffer

from src.train_d3q import (
    optimize_network,
    select_action,
    set_random_seeds,
)


# ============================================================
# CHECKPOINT EXPERIMENT SETTINGS
# ============================================================

CHECKPOINT_INTERVAL = 10

CHECKPOINT_DIR = "models/epoch_checkpoints"

RESULT_DIR = "results/epoch_analysis"


D3QN_HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "d3qn_training_per_episode.csv",
)

PPO_HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "ppo_training_per_episode.csv",
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_directories() -> None:

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )


# ============================================================
# GENERIC CSV SAVE FUNCTION
# ============================================================

def save_csv(
    path: str,
    rows: list[dict],
    fieldnames: list[str],
) -> None:

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SAVE D3QN HISTORY
# ============================================================

def save_d3qn_history(
    history: list[dict],
) -> None:

    save_csv(

        path=
            D3QN_HISTORY_PATH,

        rows=
            history,

        fieldnames=[
            "episode",
            "reward",
            "average_reward_50",
            "average_loss",
            "epsilon",
            "training_step",
        ],
    )


# ============================================================
# D3QN CHECKPOINT TRAINING
# ============================================================

def train_d3qn_checkpoints() -> None:

    print()
    print("=" * 75)
    print(
        "D3QN TRAINING WITH "
        "10-EPISODE CHECKPOINTS"
    )
    print("=" * 75)

    config = D3QN_CONFIG

    d3qn_config = config[
        "D3QN"
    ]

    # --------------------------------------------------------
    # Make sure Medium traffic is being used
    # --------------------------------------------------------

    if (
        config["LOAD_SCENARIO"]
        != "medium"
    ):
        raise ValueError(
            "Checkpoint training must use "
            "Medium traffic."
        )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    seed = int(
        config["SEED"]
    )

    set_random_seeds(
        seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = (
        NetworkSlicingD3QEnv()
    )

    state_size = int(
        config["STATE_SIZE"]
    )

    action_size = int(
        config[
            "NUMBER_OF_ACTIONS"
        ]
    )

    hidden_size = int(
        d3qn_config[
            "hidden_size"
        ]
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        env.observation_space
        .shape[0]
        != state_size
    ):
        raise ValueError(
            "D3QN state size mismatch."
        )

    if (
        env.action_space.n
        != action_size
    ):
        raise ValueError(
            "D3QN action space mismatch."
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Traffic load        : "
        f"{config['LOAD_SCENARIO']}"
    )

    print(
        f"State size          : "
        f"{state_size}"
    )

    print(
        f"Action count        : "
        f"{action_size}"
    )

    print(
        f"Total RBs           : "
        f"{config['TOTAL_RB']}"
    )

    print(
        f"Episode length      : "
        f"{config['MAX_TIME']}"
    )

    print(
        f"Training episodes   : "
        f"{d3qn_config['episodes']}"
    )

    print(
        f"Checkpoint interval : "
        f"{CHECKPOINT_INTERVAL}"
    )

    print("=" * 75)

    # ========================================================
    # ONLINE NETWORK
    # ========================================================

    online_network = (
        DuelingQNetwork(

            state_size=
                state_size,

            action_size=
                action_size,

            hidden_size=
                hidden_size,
        )
        .to(device)
    )

    # ========================================================
    # TARGET NETWORK
    # ========================================================

    target_network = (
        DuelingQNetwork(

            state_size=
                state_size,

            action_size=
                action_size,

            hidden_size=
                hidden_size,
        )
        .to(device)
    )

    target_network.load_state_dict(
        online_network.state_dict()
    )

    target_network.eval()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = Adam(

        online_network.parameters(),

        lr=float(
            d3qn_config[
                "learning_rate"
            ]
        ),
    )

    # ========================================================
    # PRIORITIZED REPLAY BUFFER
    # ========================================================

    replay_buffer = (
        PrioritizedReplayBuffer(

            capacity=int(
                d3qn_config[
                    "replay_buffer_size"
                ]
            ),

            alpha=float(
                d3qn_config[
                    "per_alpha"
                ]
            ),

            epsilon=float(
                d3qn_config[
                    "per_epsilon"
                ]
            ),
        )
    )

    # ========================================================
    # TRAINING PARAMETERS
    # ========================================================

    episodes = int(
        d3qn_config[
            "episodes"
        ]
    )

    epsilon = float(
        d3qn_config[
            "eps_start"
        ]
    )

    epsilon_end = float(
        d3qn_config[
            "eps_end"
        ]
    )

    epsilon_decay = float(
        d3qn_config[
            "eps_decay"
        ]
    )

    target_update = int(
        d3qn_config[
            "target_update"
        ]
    )

    learning_starts = int(
        d3qn_config[
            "learning_starts"
        ]
    )

    train_frequency = int(
        d3qn_config[
            "train_frequency"
        ]
    )

    batch_size = int(
        d3qn_config[
            "batch_size"
        ]
    )

    # ========================================================
    # TRAINING STATISTICS
    # ========================================================

    training_step = 0

    recent_rewards = deque(
        maxlen=50
    )

    history: list[
        dict
    ] = []

    # ========================================================
    # D3QN TRAINING LOOP
    # ========================================================

    for episode in range(
        1,
        episodes + 1,
    ):

        # Different reproducible
        # realization every episode.
        episode_seed = (
            seed + episode
        )

        state, _ = env.reset(
            seed=episode_seed
        )

        episode_reward = 0.0

        episode_losses: list[
            float
        ] = []

        terminated = False
        truncated = False

        # ====================================================
        # ONE COMPLETE EPISODE
        # ====================================================

        while not (
            terminated
            or truncated
        ):

            # ------------------------------------------------
            # Select action
            # ------------------------------------------------

            action = select_action(

                state=
                    state,

                network=
                    online_network,

                epsilon=
                    epsilon,

                action_size=
                    action_size,

                device=
                    device,
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(
                action
            )

            done = (
                terminated
                or truncated
            )

            # ------------------------------------------------
            # Store experience
            # ------------------------------------------------

            replay_buffer.add(

                state=
                    state,

                action=
                    action,

                reward=float(
                    reward
                ),

                next_state=
                    next_state,

                done=
                    done,
            )

            state = (
                next_state
            )

            episode_reward += float(
                reward
            )

            training_step += 1

            # ------------------------------------------------
            # Minimum replay size
            # ------------------------------------------------

            minimum_buffer_size = max(

                learning_starts,

                batch_size,
            )

            # ------------------------------------------------
            # Optimize network
            # ------------------------------------------------

            if (
                len(
                    replay_buffer
                )
                >= minimum_buffer_size

                and

                training_step
                % train_frequency
                == 0
            ):

                loss = optimize_network(

                    online_network=
                        online_network,

                    target_network=
                        target_network,

                    replay_buffer=
                        replay_buffer,

                    optimizer=
                        optimizer,

                    device=
                        device,

                    training_step=
                        training_step,
                )

                episode_losses.append(
                    loss
                )

        # ====================================================
        # END OF EPISODE
        # ====================================================

        # ----------------------------------------------------
        # Update epsilon
        # ----------------------------------------------------

        epsilon = max(

            epsilon_end,

            epsilon
            * epsilon_decay,
        )

        # ----------------------------------------------------
        # Reward history
        # ----------------------------------------------------

        recent_rewards.append(
            episode_reward
        )

        # ----------------------------------------------------
        # Update target network
        # ----------------------------------------------------

        if (
            episode
            % target_update
            == 0
        ):

            target_network.load_state_dict(
                online_network.state_dict()
            )

        # ----------------------------------------------------
        # Calculate averages
        # ----------------------------------------------------

        average_reward_50 = float(
            np.mean(
                recent_rewards
            )
        )

        average_loss = (

            float(
                np.mean(
                    episode_losses
                )
            )

            if episode_losses

            else 0.0
        )

        # ----------------------------------------------------
        # Save episode result
        # ----------------------------------------------------

        history.append(
            {
                "episode":
                    episode,

                "reward":
                    episode_reward,

                "average_reward_50":
                    average_reward_50,

                "average_loss":
                    average_loss,

                "epsilon":
                    epsilon,

                "training_step":
                    training_step,
            }
        )

        # Save after every episode.
        # If training is interrupted,
        # completed history is preserved.
        save_d3qn_history(
            history
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"D3QN Episode "
            f"{episode:3d}/"
            f"{episodes} | "
            f"Reward="
            f"{episode_reward:8.3f} | "
            f"Avg50="
            f"{average_reward_50:8.3f} | "
            f"Loss="
            f"{average_loss:.6f} | "
            f"Epsilon="
            f"{epsilon:.4f}"
        )

        # ====================================================
        # SAVE CHECKPOINT EVERY 10 EPISODES
        # ====================================================

        if (
            episode
            % CHECKPOINT_INTERVAL
            == 0
        ):

            checkpoint_path = (
                os.path.join(

                    CHECKPOINT_DIR,

                    (
                        f"d3qn_ep_"
                        f"{episode:03d}.pth"
                    ),
                )
            )

            torch.save(
                {
                    "episode":
                        episode,

                    "model_state_dict":
                        online_network
                        .state_dict(),

                    "target_state_dict":
                        target_network
                        .state_dict(),

                    "optimizer_state_dict":
                        optimizer
                        .state_dict(),

                    "state_size":
                        state_size,

                    "action_size":
                        action_size,

                    "hidden_size":
                        hidden_size,

                    "epsilon":
                        epsilon,

                    "training_step":
                        training_step,

                    "traffic_load":
                        config[
                            "LOAD_SCENARIO"
                        ],

                    "config":
                        config,
                },

                checkpoint_path,
            )

            print(
                f"*** D3QN checkpoint "
                f"saved: "
                f"{checkpoint_path}"
            )

    # ========================================================
    # FINISH D3QN
    # ========================================================

    env.close()

    print()

    print("=" * 75)

    print(
        "D3QN CHECKPOINT "
        "TRAINING COMPLETED"
    )

    print("=" * 75)

    print(
        f"History     : "
        f"{D3QN_HISTORY_PATH}"
    )

    print(
        f"Checkpoints : "
        f"{CHECKPOINT_DIR}"
    )


# ============================================================
# PPO CHECKPOINT CALLBACK
# ============================================================

class PPOCheckpointCallback(
    BaseCallback
):

    def __init__(
        self,
        checkpoint_interval: int,
        checkpoint_dir: str,
        history_path: str,
        expected_episodes: int,
        verbose: int = 0,
    ):

        super().__init__(
            verbose
        )

        self.checkpoint_interval = int(
            checkpoint_interval
        )

        self.checkpoint_dir = (
            checkpoint_dir
        )

        self.history_path = (
            history_path
        )

        self.expected_episodes = int(
            expected_episodes
        )

        self.episode_number = 0

        self.history: list[
            dict
        ] = []

    # ========================================================
    # SAVE PPO HISTORY
    # ========================================================

    def _save_history(
        self
    ) -> None:

        save_csv(

            path=
                self.history_path,

            rows=
                self.history,

            fieldnames=[
                "episode",
                "reward",
                "episode_length",
                "timesteps",
            ],
        )

    # ========================================================
    # CALLED AT EACH PPO STEP
    # ========================================================

    def _on_step(
        self
    ) -> bool:

        infos = self.locals.get(
            "infos",
            [],
        )

        for info in infos:

            if (
                "episode"
                not in info
            ):
                continue

            # ------------------------------------------------
            # Completed episode
            # ------------------------------------------------

            self.episode_number += 1

            reward = float(
                info[
                    "episode"
                ][
                    "r"
                ]
            )

            episode_length = int(
                info[
                    "episode"
                ][
                    "l"
                ]
            )

            # ------------------------------------------------
            # Save history
            # ------------------------------------------------

            self.history.append(
                {
                    "episode":
                        self.episode_number,

                    "reward":
                        reward,

                    "episode_length":
                        episode_length,

                    "timesteps":
                        self.num_timesteps,
                }
            )

            # Save after every episode
            self._save_history()

            print(
                f"PPO Episode "
                f"{self.episode_number:3d}/"
                f"{self.expected_episodes} | "
                f"Reward="
                f"{reward:8.3f} | "
                f"Timesteps="
                f"{self.num_timesteps}"
            )

            # =================================================
            # SAVE PPO EVERY 10 EPISODES
            # =================================================

            if (
                self.episode_number
                % self.checkpoint_interval
                == 0

                and

                self.episode_number
                <= self.expected_episodes
            ):

                checkpoint_base = (
                    os.path.join(

                        self.checkpoint_dir,

                        (
                            f"ppo_ep_"
                            f"{self.episode_number:03d}"
                        ),
                    )
                )

                self.model.save(
                    checkpoint_base
                )

                print(
                    f"*** PPO checkpoint "
                    f"saved: "
                    f"{checkpoint_base}.zip"
                )

        return True

    # ========================================================
    # END OF PPO TRAINING
    # ========================================================

    def _on_training_end(
        self
    ) -> None:

        self._save_history()

        print(
            f"PPO history saved: "
            f"{self.history_path}"
        )


# ============================================================
# PPO CHECKPOINT TRAINING
# ============================================================

def train_ppo_checkpoints() -> None:

    print()
    print("=" * 75)

    print(
        "PPO TRAINING WITH "
        "10-EPISODE CHECKPOINTS"
    )

    print("=" * 75)

    config = PPO_CONFIG

    ppo_config = config[
        "PPO"
    ]

    # --------------------------------------------------------
    # Medium traffic safety check
    # --------------------------------------------------------

    if (
        config["LOAD_SCENARIO"]
        != "medium"
    ):

        raise ValueError(
            "Checkpoint training must "
            "use Medium traffic."
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = (
        NetworkSlicingPPOEnv()
    )

    check_env(
        env,
        warn=True,
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        env.observation_space
        .shape[0]
        != int(
            config[
                "STATE_SIZE"
            ]
        )
    ):

        raise ValueError(
            "PPO state size mismatch."
        )

    if (
        env.action_space.n
        != int(
            config[
                "NUMBER_OF_ACTIONS"
            ]
        )
    ):

        raise ValueError(
            "PPO action space mismatch."
        )

    # --------------------------------------------------------
    # Monitor wrapper
    # --------------------------------------------------------

    env = Monitor(
        env
    )

    # --------------------------------------------------------
    # Expected episodes
    # --------------------------------------------------------

    expected_episodes = int(

        ppo_config[
            "total_timesteps"
        ]

        / config[
            "MAX_TIME"
        ]
    )

    # --------------------------------------------------------
    # Print setup
    # --------------------------------------------------------

    print(
        f"Traffic load        : "
        f"{config['LOAD_SCENARIO']}"
    )

    print(
        f"State size          : "
        f"{config['STATE_SIZE']}"
    )

    print(
        f"Action count        : "
        f"{config['NUMBER_OF_ACTIONS']}"
    )

    print(
        f"Total RBs           : "
        f"{config['TOTAL_RB']}"
    )

    print(
        f"Episode length      : "
        f"{config['MAX_TIME']}"
    )

    print(
        f"Total timesteps     : "
        f"{ppo_config['total_timesteps']}"
    )

    print(
        f"Expected episodes   : "
        f"{expected_episodes}"
    )

    print(
        f"Checkpoint interval : "
        f"{CHECKPOINT_INTERVAL}"
    )

    print("=" * 75)

    # ========================================================
    # PPO MODEL
    # ========================================================

    model = PPO(

        policy=
            "MlpPolicy",

        env=
            env,

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
            config[
                "SEED"
            ]
        ),

        verbose=0,

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

    # ========================================================
    # CALLBACK
    # ========================================================

    callback = (
        PPOCheckpointCallback(

            checkpoint_interval=
                CHECKPOINT_INTERVAL,

            checkpoint_dir=
                CHECKPOINT_DIR,

            history_path=
                PPO_HISTORY_PATH,

            expected_episodes=
                expected_episodes,
        )
    )

    # ========================================================
    # TRAIN PPO
    # ========================================================

    model.learn(

        total_timesteps=int(
            ppo_config[
                "total_timesteps"
            ]
        ),

        callback=
            callback,

        progress_bar=False,
    )

    # ========================================================
    # FINISH PPO
    # ========================================================

    env.close()

    print()

    print("=" * 75)

    print(
        "PPO CHECKPOINT "
        "TRAINING COMPLETED"
    )

    print("=" * 75)

    print(
        f"History     : "
        f"{PPO_HISTORY_PATH}"
    )

    print(
        f"Checkpoints : "
        f"{CHECKPOINT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    create_output_directories()

    # --------------------------------------------------------
    # Allows us to train D3QN and PPO separately.
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(

        description=(
            "Train D3QN/PPO and save "
            "checkpoints every 10 episodes."
        )
    )

    parser.add_argument(

        "--method",

        choices=[
            "d3qn",
            "ppo",
            "both",
        ],

        default=
            "both",

        help=(
            "Choose d3qn, ppo, "
            "or both."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("=" * 75)

    print(
        "PER-EPISODE "
        "CHECKPOINT TRAINING"
    )

    print("=" * 75)

    print(
        "Existing final models "
        "are NOT overwritten."
    )

    print(
        f"Checkpoint folder : "
        f"{CHECKPOINT_DIR}"
    )

    print(
        f"Results folder    : "
        f"{RESULT_DIR}"
    )

    print(
        f"Selected method   : "
        f"{args.method}"
    )

    print("=" * 75)

    # ========================================================
    # D3QN
    # ========================================================

    if args.method in {
        "d3qn",
        "both",
    }:

        train_d3qn_checkpoints()

    # ========================================================
    # PPO
    # ========================================================

    if args.method in {
        "ppo",
        "both",
    }:

        train_ppo_checkpoints()

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 75)

    print(
        "REQUESTED CHECKPOINT "
        "TRAINING FINISHED"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()