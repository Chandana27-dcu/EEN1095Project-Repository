from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for Double Dueling DQN.

    Experiences with larger TD errors are sampled more frequently.
    Importance-sampling weights reduce the bias introduced by
    prioritized sampling.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-5,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if epsilon <= 0.0:
            raise ValueError("epsilon must be greater than zero.")

        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

        self.buffer: list[Experience | None] = [
            None
            for _ in range(self.capacity)
        ]

        self.priorities = np.zeros(
            self.capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one transition to the replay buffer.

        New transitions receive the current maximum priority so they
        have a good chance of being sampled at least once.
        """

        state_array = np.asarray(
            state,
            dtype=np.float32,
        ).copy()

        next_state_array = np.asarray(
            next_state,
            dtype=np.float32,
        ).copy()

        experience = Experience(
            state=state_array,
            action=int(action),
            reward=float(reward),
            next_state=next_state_array,
            done=bool(done),
        )

        self.buffer[self.position] = experience

        if self.size == 0:
            max_priority = 1.0
        else:
            max_priority = float(
                np.max(self.priorities[:self.size])
            )

            if max_priority <= 0.0:
                max_priority = 1.0

        self.priorities[self.position] = max_priority

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> dict[str, Any]:
        """
        Sample a prioritized mini-batch.

        Returns:
            states
            actions
            rewards
            next_states
            dones
            indices
            weights
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.size < batch_size:
            raise ValueError(
                f"Not enough experiences to sample. "
                f"Buffer size={self.size}, batch_size={batch_size}."
            )

        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                "beta must be between 0 and 1."
            )

        current_priorities = self.priorities[:self.size]

        scaled_priorities = np.power(
            current_priorities + self.epsilon,
            self.alpha,
        )

        priority_sum = float(
            np.sum(scaled_priorities)
        )

        if priority_sum <= 0.0:
            probabilities = np.full(
                self.size,
                1.0 / self.size,
                dtype=np.float32,
            )
        else:
            probabilities = (
                scaled_priorities / priority_sum
            )

        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        experiences = [
            self.buffer[index]
            for index in indices
        ]

        if any(
            experience is None
            for experience in experiences
        ):
            raise RuntimeError(
                "Sampled an empty replay-buffer entry."
            )

        sampled_probabilities = probabilities[indices]

        weights = np.power(
            self.size * sampled_probabilities,
            -beta,
        )

        weights = weights / max(
            float(np.max(weights)),
            self.epsilon,
        )

        typed_experiences = [
            experience
            for experience in experiences
            if experience is not None
        ]

        states = np.stack(
            [
                experience.state
                for experience in typed_experiences
            ]
        ).astype(np.float32)

        actions = np.asarray(
            [
                experience.action
                for experience in typed_experiences
            ],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                experience.reward
                for experience in typed_experiences
            ],
            dtype=np.float32,
        )

        next_states = np.stack(
            [
                experience.next_state
                for experience in typed_experiences
            ]
        ).astype(np.float32)

        dones = np.asarray(
            [
                experience.done
                for experience in typed_experiences
            ],
            dtype=np.float32,
        )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": indices.astype(np.int64),
            "weights": weights.astype(np.float32),
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """
        Update priorities after calculating TD errors.
        """

        indices_array = np.asarray(
            indices,
            dtype=np.int64,
        )

        errors_array = np.asarray(
            td_errors,
            dtype=np.float32,
        )

        if indices_array.shape != errors_array.shape:
            raise ValueError(
                "indices and td_errors must have matching shapes."
            )

        for index, td_error in zip(
            indices_array,
            errors_array,
        ):
            if not 0 <= int(index) < self.size:
                raise IndexError(
                    f"Replay index {index} is out of range."
                )

            new_priority = (
                abs(float(td_error))
                + self.epsilon
            )

            self.priorities[int(index)] = new_priority

    def clear(self) -> None:
        """
        Remove all stored experiences.
        """

        self.buffer = [
            None
            for _ in range(self.capacity)
        ]

        self.priorities.fill(0.0)
        self.position = 0
        self.size = 0