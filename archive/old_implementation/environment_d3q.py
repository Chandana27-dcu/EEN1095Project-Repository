from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.actions_d3q import ACTIONS, get_rb_allocation
from src.config_d3q import CONFIG
from src.traffic_d3q import generate_traffic


class NetworkSlicingD3QEnv(gym.Env):
    """
    Double Dueling DQN environment for network-slice resource allocation.

    Slices:
        - eMBB
        - URLLC1
        - URLLC2
        - BE1

    D3QN action:
        One integer action index from 0 to 154.

    Each action index selects one of 155 predefined percentage-based
    resource-allocation settings. Every allocation:

        - uses 5% increments;
        - provides each slice with at least the configured minimum share;
        - adds up to exactly 100%.

    State values per slice:
        1. Throughput satisfaction
        2. Latency
        3. PLR
        4. Queue occupancy
        5. Channel condition
        6. Current traffic load

    Total observation size:
        4 slices × 6 values = 24 values
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------
        # Main configuration
        # --------------------------------------------------

        self.slices = list(CONFIG["SLICES"])
        self.number_of_slices = len(self.slices)

        if len(ACTIONS) != int(CONFIG["NUMBER_OF_ACTIONS"]):
            raise ValueError(
                "The generated D3QN action count does not match "
                "NUMBER_OF_ACTIONS."
            )

        self.total_rb = float(CONFIG["TOTAL_RB"])
        self.bits_per_rb = float(CONFIG["BITS_PER_RB"])
        self.max_time = int(CONFIG["MAX_TIME"])
        self.buffer_size = int(CONFIG["BUFFER_SIZE"])
        self.queue_reference = float(CONFIG.get("QUEUE_REFERENCE", 500.0))

        self.slot_duration_ms = float(CONFIG["SLOT_DURATION_MS"])
        self.minimum_share = float(
            CONFIG["MINIMUM_SHARE_PERCENT"]
        ) / 100.0

        self.throughput_window = int(CONFIG["THROUGHPUT_WINDOW"])
        self.latency_window = int(CONFIG["LATENCY_WINDOW"])

        self.qos = CONFIG["QOS"]
        self.deadline_ms = CONFIG["DEADLINE_MS"]

        self.reward_weights = CONFIG["REWARD_WEIGHTS"]
        self.reward_penalties = CONFIG.get("REWARD_PENALTIES", {})

        # --------------------------------------------------
        # Channel configuration
        # --------------------------------------------------

        channel_config = CONFIG["CHANNEL"]

        self.channel_min = float(channel_config["min"])
        self.channel_max = float(channel_config["max"])
        self.channel_correlation = float(
            channel_config["correlation"]
        )
        self.channel_mean = float(channel_config["mean"])
        self.channel_noise_std = float(
            channel_config["noise_std"]
        )

        # --------------------------------------------------
        # D3QN action space: one of 155 predefined allocations.
        # --------------------------------------------------
        self.action_space = spaces.Discrete(len(ACTIONS))

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------

        # All observation values are normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.number_of_slices * 6,),
            dtype=np.float32,
        )

        self.time = 0

        self.queue: dict[str, list[dict[str, Any]]] = {}
        self.metrics: dict[str, dict[str, Any]] = {}

        self.last_action = np.full(
            self.number_of_slices,
            1.0 / self.number_of_slices,
            dtype=np.float32,
        )

        self.reset()

    # ======================================================
    # Gymnasium API
    # ======================================================

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.time = 0

        self.queue = {
            slice_name: []
            for slice_name in self.slices
        }

        self.metrics = {
            # Moving-average throughput satisfaction.
            "throughput_percent": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Throughput percentage for each simulation step.
            "step_throughput_percent": {
                slice_name: []
                for slice_name in self.slices
            },

            # Offered traffic bits in the current step.
            "step_offered_bits": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Successfully transmitted bits.
            "throughput_bits": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Latency of successfully served packets, in ms.
            "latency_ms": {
                slice_name: []
                for slice_name in self.slices
            },

            # Packet loss ratio, in percentage.
            "plr_percent": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Total generated packets.
            "arrivals": {
                slice_name: 0
                for slice_name in self.slices
            },

            # Total dropped packets.
            "dropped": {
                slice_name: 0
                for slice_name in self.slices
            },

            # Current channel quality.
            "channel": {
                slice_name: self.channel_mean
                for slice_name in self.slices
            },

            # Packets generated in the current step.
            "recent_arrivals": {
                slice_name: 0
                for slice_name in self.slices
            },
        }

        self.last_action = np.full(
            self.number_of_slices,
            1.0 / self.number_of_slices,
            dtype=np.float32,
        )

        observation = self.get_state()

        return observation, {}

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        self.time += 1

        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(
                f"Invalid D3QN action index: {action_index}"
            )

        selected_action = ACTIONS[action_index]
        allocation_shares = np.asarray(
            [
                selected_action[slice_name] / 100.0
                for slice_name in self.slices
            ],
            dtype=np.float32,
        )
        self.last_action = allocation_shares.copy()

        allocation_percent = {
            slice_name: float(selected_action[slice_name])
            for slice_name in self.slices
        }

        rb_allocation = get_rb_allocation(
            action_index=action_index,
            total_rbs=self.total_rb,
        )

        # Update channel conditions before transmission.
        self._update_channel_conditions()

        # Generate new packets.
        generated_traffic = generate_traffic(
            time_step=self.time,
            rng=self.np_random,
        )

        for slice_name in self.slices:
            new_packets = generated_traffic[slice_name]

            offered_bits = sum(
                float(packet["size"])
                for packet in new_packets
            )

            self.metrics["step_offered_bits"][
                slice_name
            ] = offered_bits

            self.metrics["recent_arrivals"][
                slice_name
            ] = len(new_packets)

            for packet in new_packets:
                self.queue[slice_name].append(packet)
                self.metrics["arrivals"][slice_name] += 1

        step_throughput_bits = {
            slice_name: 0.0
            for slice_name in self.slices
        }

        # Serve packets in every slice.
        for slice_name in self.slices:
            self._serve_slice_packets(
                slice_name=slice_name,
                allocated_rbs=rb_allocation[slice_name],
                step_throughput_bits=step_throughput_bits,
            )

        self._update_metrics(step_throughput_bits)

        reward = self._calculate_reward(
            allocation_shares
        )

        terminated = self.time >= self.max_time
        truncated = False

        observation = self.get_state()

        info = {
            "action_index": action_index,
            "allocation_percent": allocation_percent,
            "rb_allocation": rb_allocation,
            "metrics": self.get_episode_metrics(),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # Packet service
    # ======================================================

    def _serve_slice_packets(
        self,
        slice_name: str,
        allocated_rbs: float,
        step_throughput_bits: dict[str, float],
    ) -> None:
        channel_factor = float(
            self.metrics["channel"][slice_name]
        )

        capacity_bits = (
            allocated_rbs
            * self.bits_per_rb
            * channel_factor
        )

        remaining_queue: list[dict[str, Any]] = []

        # FIFO packet transmission.
        for packet in self.queue[slice_name]:
            packet_size = float(packet["size"])

            if packet_size <= capacity_bits:
                # Minimum latency is one simulation slot.
                waiting_slots = max(
                    1,
                    self.time - int(packet["arrival"]),
                )

                latency_ms = (
                    waiting_slots
                    * self.slot_duration_ms
                )

                self.metrics["latency_ms"][
                    slice_name
                ].append(float(latency_ms))

                self.metrics["throughput_bits"][
                    slice_name
                ] += packet_size

                step_throughput_bits[
                    slice_name
                ] += packet_size

                capacity_bits -= packet_size
            else:
                remaining_queue.append(packet)

        # Drop expired packets.
        valid_packets: list[dict[str, Any]] = []

        deadline_ms = float(
            self.deadline_ms[slice_name]
        )

        for packet in remaining_queue:
            waiting_slots = max(
                1,
                self.time - int(packet["arrival"]),
            )

            waiting_time_ms = (
                waiting_slots
                * self.slot_duration_ms
            )

            if waiting_time_ms > deadline_ms:
                self.metrics["dropped"][
                    slice_name
                ] += 1
            else:
                valid_packets.append(packet)

        remaining_queue = valid_packets

        # Apply queue buffer limit.
        if len(remaining_queue) > self.buffer_size:
            overflow_count = (
                len(remaining_queue)
                - self.buffer_size
            )

            self.metrics["dropped"][
                slice_name
            ] += overflow_count

            # Keep oldest packets for FIFO behaviour.
            remaining_queue = remaining_queue[
                : self.buffer_size
            ]

        self.queue[slice_name] = remaining_queue

    # ======================================================
    # Channel model
    # ======================================================

    def _update_channel_conditions(self) -> None:
        """
        Update channel conditions using a correlated random process.

        This is more realistic than independently selecting a completely
        new channel condition at each time step.
        """

        for slice_name in self.slices:
            previous_channel = float(
                self.metrics["channel"][
                    slice_name
                ]
            )

            channel_noise = float(
                self.np_random.normal(
                    loc=0.0,
                    scale=self.channel_noise_std,
                )
            )

            new_channel = (
                self.channel_correlation
                * previous_channel
                + (
                    1.0
                    - self.channel_correlation
                )
                * self.channel_mean
                + channel_noise
            )

            self.metrics["channel"][
                slice_name
            ] = float(
                np.clip(
                    new_channel,
                    self.channel_min,
                    self.channel_max,
                )
            )

    # ======================================================
    # Metrics
    # ======================================================

    def _update_metrics(
        self,
        step_throughput_bits: dict[str, float],
    ) -> None:
        for slice_name in self.slices:
            offered_bits = float(
                self.metrics["step_offered_bits"][slice_name]
            )
            served_bits = float(
                step_throughput_bits[slice_name]
            )

            if offered_bits > 0.0:
                throughput_percent = (
                    served_bits / offered_bits
                ) * 100.0
            else:
                throughput_percent = (
                    100.0
                    if len(self.queue[slice_name]) == 0
                    else 0.0
                )

            throughput_percent = float(
                np.clip(
                    throughput_percent,
                    0.0,
                    100.0,
                )
            )

            throughput_history = self.metrics[
                "step_throughput_percent"
            ][slice_name]

            throughput_history.append(
                throughput_percent
            )

            recent_throughput = (
                throughput_history[
                    -self.throughput_window:
                ]
            )

            self.metrics["throughput_percent"][
                slice_name
            ] = float(
                np.mean(recent_throughput)
                if recent_throughput
                else 0.0
            )

            arrivals = int(
                self.metrics["arrivals"][
                    slice_name
                ]
            )

            dropped = int(
                self.metrics["dropped"][
                    slice_name
                ]
            )

            if arrivals > 0:
                plr_percent = (
                    dropped / arrivals
                ) * 100.0
            else:
                plr_percent = 0.0

            self.metrics["plr_percent"][
                slice_name
            ] = float(
                np.clip(
                    plr_percent,
                    0.0,
                    100.0,
                )
            )

    # ======================================================
    # Reward function
    # ======================================================

    def _calculate_reward(
        self,
        action: np.ndarray,
    ) -> float:
        """
        Calculate one normalized reward for all four slices.

        For every slice, the reward includes:
            - throughput score in the range 0 to 1;
            - latency score derived from latency in milliseconds;
            - PLR score derived from PLR percentage.

        The final reward is the mean of the four slice rewards.

        Not used:
            - global penalty;
            - starvation penalty;
            - Jain fairness;
            - worst-slice protection.
        """

        del action

        throughput_weight = float(
            self.reward_weights["throughput"]
        )
        latency_weight = float(
            self.reward_weights["latency"]
        )
        plr_weight = float(
            self.reward_weights["plr"]
        )

        weight_sum = (
            throughput_weight
            + latency_weight
            + plr_weight
        )

        if not np.isclose(weight_sum, 1.0):
            raise ValueError(
                "Throughput, latency and PLR reward weights "
                f"must sum to 1.0, but sum to {weight_sum:.4f}."
            )

        slice_rewards: list[float] = []

        for slice_name in self.slices:
            throughput_percent = float(
                self.metrics["throughput_percent"][slice_name]
            )
            throughput_score = float(
                np.clip(
                    throughput_percent / 100.0,
                    0.0,
                    1.0,
                )
            )

            latency_history = self.metrics["latency_ms"][slice_name]
            recent_latencies = latency_history[-self.latency_window:]
            latency_requirement_ms = float(
                self.qos[slice_name]["latency_req_ms"]
            )

            if recent_latencies:
                average_latency_ms = float(np.mean(recent_latencies))
            else:
                average_latency_ms = latency_requirement_ms

            latency_score = float(
                np.clip(
                    1.0
                    - average_latency_ms
                    / max(latency_requirement_ms, 1e-8),
                    0.0,
                    1.0,
                )
            )

            plr_percent = float(
                self.metrics["plr_percent"][slice_name]
            )
            plr_requirement_percent = float(
                self.qos[slice_name]["plr_req"]
            )
            plr_score = float(
                np.clip(
                    1.0
                    - plr_percent
                    / max(plr_requirement_percent, 1e-8),
                    0.0,
                    1.0,
                )
            )

            slice_reward = (
                throughput_weight * throughput_score
                + latency_weight * latency_score
                + plr_weight * plr_score
            )

            slice_rewards.append(
                float(np.clip(slice_reward, 0.0, 1.0))
            )

        total_reward = float(np.mean(slice_rewards))

        return float(np.clip(total_reward, 0.0, 1.0))

    # ======================================================
    # State representation
    # ======================================================

    def get_state(self) -> np.ndarray:
        """
        Return a normalized 24-value state vector.

        Per slice:
            throughput, latency, PLR, queue, channel, traffic load
        """

        state: list[float] = []

        for slice_name in self.slices:
            # ----------------------------------------------
            # Throughput
            # ----------------------------------------------

            throughput_normalized = float(
                np.clip(
                    self.metrics[
                        "throughput_percent"
                    ][slice_name]
                    / 100.0,
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Latency
            # ----------------------------------------------

            latency_history = self.metrics[
                "latency_ms"
            ][slice_name]

            recent_latencies = latency_history[
                -self.latency_window:
            ]

            average_latency_ms = float(
                np.mean(recent_latencies)
                if recent_latencies
                else 0.0
            )

            required_latency_ms = float(
                self.qos[slice_name][
                    "latency_req_ms"
                ]
            )

            latency_normalized = float(
                np.clip(
                    average_latency_ms
                    / max(
                        required_latency_ms
                        * 5.0,
                        1e-8,
                    ),
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # PLR
            # ----------------------------------------------

            plr_normalized = float(
                np.clip(
                    self.metrics["plr_percent"][
                        slice_name
                    ]
                    / 100.0,
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Queue occupancy
            # ----------------------------------------------

            queue_normalized = float(
                np.clip(
                    len(self.queue[slice_name])
                    / max(
                        self.queue_reference,
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Channel condition
            # ----------------------------------------------

            channel_value = float(
                self.metrics["channel"][
                    slice_name
                ]
            )

            channel_normalized = float(
                np.clip(
                    (
                        channel_value
                        - self.channel_min
                    )
                    / max(
                        self.channel_max
                        - self.channel_min,
                        1e-8,
                    ),
                    0.0,
                    1.0,
                )
            )

            # ----------------------------------------------
            # Traffic load
            # ----------------------------------------------

            recent_arrivals = int(
                self.metrics[
                    "recent_arrivals"
                ][slice_name]
            )

            traffic_load_normalized = float(
                np.clip(
                    recent_arrivals / 20.0,
                    0.0,
                    1.0,
                )
            )

            state.extend(
                [
                    throughput_normalized,
                    latency_normalized,
                    plr_normalized,
                    queue_normalized,
                    channel_normalized,
                    traffic_load_normalized,
                ]
            )

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        return np.clip(
            state_array,
            0.0,
            1.0,
        ).astype(np.float32)

    # ======================================================
    # Human-readable evaluation metrics
    # ======================================================

    def get_episode_metrics(
        self,
    ) -> dict[str, dict[str, float | int]]:
        results: dict[
            str,
            dict[str, float | int],
        ] = {}

        for slice_name in self.slices:
            latency_history = self.metrics[
                "latency_ms"
            ][slice_name]

            if latency_history:
                average_latency_ms = float(
                    np.mean(latency_history)
                )

                jitter_ms = float(
                    np.std(latency_history)
                    if len(latency_history) > 1
                    else 0.0
                )
            else:
                average_latency_ms = 0.0
                jitter_ms = 0.0

            throughput_history = self.metrics[
                "step_throughput_percent"
            ][slice_name]

            average_throughput_percent = float(
                np.mean(throughput_history)
                if throughput_history
                else 0.0
            )

            results[slice_name] = {
                "average_throughput_percent":
                    average_throughput_percent,

                "current_throughput_percent":
                    float(
                        self.metrics[
                            "throughput_percent"
                        ][slice_name]
                    ),

                # Latency is reported only in milliseconds.
                "average_latency_ms":
                    average_latency_ms,

                "jitter_ms":
                    jitter_ms,

                "plr_percent":
                    float(
                        self.metrics["plr_percent"][
                            slice_name
                        ]
                    ),

                "queue_length":
                    len(self.queue[slice_name]),

                "channel_condition":
                    float(
                        self.metrics["channel"][
                            slice_name
                        ]
                    ),

                "arrivals":
                    int(
                        self.metrics["arrivals"][
                            slice_name
                        ]
                    ),

                "dropped":
                    int(
                        self.metrics["dropped"][
                            slice_name
                        ]
                    ),

                "total_throughput_bits":
                    float(
                        self.metrics[
                            "throughput_bits"
                        ][slice_name]
                    ),

                "final_allocation_percent":
                    float(
                        self.last_action[
                            self.slices.index(
                                slice_name
                            )
                        ]
                        * 100.0
                    ),
            }

        return results