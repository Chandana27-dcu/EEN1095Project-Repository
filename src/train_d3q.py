from __future__ import annotations

import csv
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


# ============================================================
# D3QN TRAINING PATHS
# ============================================================

LOAD_SCENARIO = CONFIG["LOAD_SCENARIO"]

MODEL_PATH = CONFIG["MODEL_PATH"]

REWARD_NPY_PATH = (
    f"results/d3qn_{LOAD_SCENARIO}_training_rewards.npy"
)

LOSS_NPY_PATH = (
    f"results/d3qn_{LOAD_SCENARIO}_training_losses.npy"
)

TRAINING_CSV_PATH = (
    f"results/d3qn_{LOAD_SCENARIO}_training_history.csv"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

def set_random_seeds(
    seed: int,
) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PER BETA SCHEDULE
# ============================================================

def calculate_beta(
    training_step: int,
) -> float:
    """
    Increase Prioritized Experience Replay beta
    gradually from beta_start to 1.0.
    """

    d3qn_config = CONFIG["D3QN"]

    beta_start = float(
        d3qn_config["per_beta_start"]
    )

    beta_frames = int(
        d3qn_config["per_beta_frames"]
    )

    progress = min(
        training_step
        / max(
            beta_frames,
            1,
        ),
        1.0,
    )

    beta = (
        beta_start
        + progress
        * (
            1.0
            - beta_start
        )
    )

    return float(beta)


# ============================================================
# ACTION SELECTION
# ============================================================

def select_action(
    state: np.ndarray,
    network: DuelingQNetwork,
    epsilon: float,
    action_size: int,
    device: torch.device,
) -> int:
    """
    Select one discrete action using
    epsilon-greedy exploration.
    """

    # --------------------------------------------------------
    # Exploration
    # --------------------------------------------------------

    if random.random() < epsilon:
        return random.randrange(
            action_size
        )

    # --------------------------------------------------------
    # Exploitation
    # --------------------------------------------------------

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = network(
            state_tensor
        )

    action_index = int(
        torch.argmax(
            q_values,
            dim=1,
        ).item()
    )

    return action_index


# ============================================================
# NETWORK OPTIMIZATION
# ============================================================

def optimize_network(
    online_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    replay_buffer: PrioritizedReplayBuffer,
    optimizer: Adam,
    device: torch.device,
    training_step: int,
) -> float:
    """
    Perform one Double DQN optimization step
    using Prioritized Experience Replay.
    """

    d3qn_config = CONFIG["D3QN"]

    batch_size = int(
        d3qn_config["batch_size"]
    )

    gamma = float(
        d3qn_config["gamma"]
    )

    max_grad_norm = float(
        d3qn_config[
            "max_grad_norm"
        ]
    )

    beta = calculate_beta(
        training_step
    )

    # --------------------------------------------------------
    # Sample prioritized replay batch
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Current Q values
    # --------------------------------------------------------

    current_q_values = (
        online_network(
            states
        )
        .gather(
            1,
            actions,
        )
        .squeeze(1)
    )

    # --------------------------------------------------------
    # Double-DQN target
    # --------------------------------------------------------

    with torch.no_grad():

        # Online network chooses next action.
        next_action_indices = (
            online_network(
                next_states
            )
            .argmax(
                dim=1,
                keepdim=True,
            )
        )

        # Target network evaluates chosen action.
        next_q_values = (
            target_network(
                next_states
            )
            .gather(
                1,
                next_action_indices,
            )
            .squeeze(1)
        )

        target_q_values = (
            rewards
            + gamma
            * next_q_values
            * (
                1.0
                - dones
            )
        )

    # --------------------------------------------------------
    # TD error
    # --------------------------------------------------------

    td_errors = (
        target_q_values
        - current_q_values
    )

    # --------------------------------------------------------
    # Huber loss
    # --------------------------------------------------------

    element_losses = (
        nn.functional.smooth_l1_loss(
            current_q_values,
            target_q_values,
            reduction="none",
        )
    )

    loss = torch.mean(
        importance_weights
        * element_losses
    )

    # --------------------------------------------------------
    # Gradient update
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    nn.utils.clip_grad_norm_(
        online_network.parameters(),
        max_grad_norm,
    )

    optimizer.step()

    # --------------------------------------------------------
    # Update replay priorities
    # --------------------------------------------------------

    replay_buffer.update_priorities(
        indices=
            batch["indices"],

        td_errors=
            td_errors
            .detach()
            .cpu()
            .numpy(),
    )

    return float(
        loss.item()
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(
    history: list[dict[str, float | int]],
) -> None:
    """
    Save episode-level training data to CSV.

    This CSV will later be used for the professor-requested
    D3QN/PPO convergence plot.
    """

    if not history:
        return

    with open(
        TRAINING_CSV_PATH,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        field_names = [
            "episode",
            "reward",
            "average_reward_50",
            "average_loss",
            "epsilon",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# TRAIN D3QN
# ============================================================

def train_d3q() -> None:
    """
    Train the D3QN agent using the corrected common
    network-slicing environment.
    """

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    d3qn_config = CONFIG[
        "D3QN"
    ]

    seed = int(
        CONFIG["SEED"]
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

    print("=" * 70)
    print("D3QN TRAINING")
    print("=" * 70)

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Traffic load        : "
        f"{LOAD_SCENARIO}"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = NetworkSlicingD3QEnv()

    state_size = int(
        CONFIG["STATE_SIZE"]
    )

    action_size = int(
        CONFIG["NUMBER_OF_ACTIONS"]
    )

    # Important safety checks.
    if (
        env.observation_space.shape[0]
        != state_size
    ):
        raise ValueError(
            "Environment state size does not "
            "match CONFIG['STATE_SIZE']."
        )

    if (
        env.action_space.n
        != action_size
    ):
        raise ValueError(
            "Environment action count does not "
            "match CONFIG['NUMBER_OF_ACTIONS']."
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
        f"{CONFIG['TOTAL_RB']}"
    )

    # --------------------------------------------------------
    # D3QN network settings
    # --------------------------------------------------------

    hidden_size = int(
        d3qn_config[
            "hidden_size"
        ]
    )

    learning_rate = float(
        d3qn_config[
            "learning_rate"
        ]
    )

    # --------------------------------------------------------
    # Online network
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Target network
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        online_network.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------------
    # Prioritized replay buffer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    training_step = 0

    episode_rewards: list[
        float
    ] = []

    training_losses: list[
        float
    ] = []

    recent_rewards: deque[
        float
    ] = deque(
        maxlen=50
    )

    training_history: list[
        dict[str, float | int]
    ] = []

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True,
    )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    print(
        f"Training episodes   : "
        f"{episodes}"
    )

    print(
        f"Episode length      : "
        f"{CONFIG['MAX_TIME']} steps"
    )

    print(
        f"Model output        : "
        f"{MODEL_PATH}"
    )

    print("=" * 70)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(
        1,
        episodes + 1,
    ):

        # Different but reproducible traffic/channel
        # realization for each training episode.
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

        while not (
            terminated
            or truncated
        ):

            # ------------------------------------------------
            # Choose action
            # ------------------------------------------------

            action = select_action(
                state=state,
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
            # Store transition
            # ------------------------------------------------

            replay_buffer.add(
                state=
                    state,

                action=
                    action,

                reward=
                    float(
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

            episode_reward += (
                float(
                    reward
                )
            )

            training_step += 1

            # ------------------------------------------------
            # Start optimization once enough data exists
            # ------------------------------------------------

            minimum_buffer_size = max(
                learning_starts,
                batch_size,
            )

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

                training_losses.append(
                    loss
                )

        # ====================================================
        # END OF EPISODE
        # ====================================================

        epsilon = max(
            epsilon_end,
            epsilon
            * epsilon_decay,
        )

        episode_rewards.append(
            episode_reward
        )

        recent_rewards.append(
            episode_reward
        )

        # ----------------------------------------------------
        # Target network update
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
        # Statistics
        # ----------------------------------------------------

        average_recent_reward = float(
            np.mean(
                recent_rewards
            )
        )

        average_episode_loss = (
            float(
                np.mean(
                    episode_losses
                )
            )
            if episode_losses
            else 0.0
        )

        training_history.append(
            {
                "episode":
                    episode,

                "reward":
                    episode_reward,

                "average_reward_50":
                    average_recent_reward,

                "average_loss":
                    average_episode_loss,

                "epsilon":
                    epsilon,
            }
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"Episode "
            f"{episode:4d}/"
            f"{episodes} | "

            f"Reward="
            f"{episode_reward:8.3f} | "

            f"Average50="
            f"{average_recent_reward:8.3f} | "

            f"Loss="
            f"{average_episode_loss:8.5f} | "

            f"Epsilon="
            f"{epsilon:.4f} | "

            f"Buffer="
            f"{len(replay_buffer)}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            episode
            % 100
            == 0
        ):

            checkpoint_path = (
                f"models/"
                f"d3qn_"
                f"{LOAD_SCENARIO}_"
                f"checkpoint_"
                f"{episode}.pth"
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

                    "epsilon":
                        epsilon,

                    "training_step":
                        training_step,

                    "traffic_load":
                        LOAD_SCENARIO,

                    "config":
                        CONFIG,
                },
                checkpoint_path,
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                online_network
                .state_dict(),

            "target_state_dict":
                target_network
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
                LOAD_SCENARIO,

            "config":
                CONFIG,
        },
        MODEL_PATH,
    )

    # ========================================================
    # SAVE REWARDS
    # ========================================================

    np.save(
        REWARD_NPY_PATH,
        np.asarray(
            episode_rewards,
            dtype=np.float32,
        ),
    )

    # ========================================================
    # SAVE LOSSES
    # ========================================================

    np.save(
        LOSS_NPY_PATH,
        np.asarray(
            training_losses,
            dtype=np.float32,
        ),
    )

    # ========================================================
    # SAVE CSV FOR CONVERGENCE GRAPH
    # ========================================================

    save_training_history(
        training_history
    )

    env.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("D3QN TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Traffic load : "
        f"{LOAD_SCENARIO}"
    )

    print(
        f"Model saved  : "
        f"{MODEL_PATH}"
    )

    print(
        f"Rewards NPY  : "
        f"{REWARD_NPY_PATH}"
    )

    print(
        f"Losses NPY   : "
        f"{LOSS_NPY_PATH}"
    )

    print(
        f"History CSV  : "
        f"{TRAINING_CSV_PATH}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_d3q()