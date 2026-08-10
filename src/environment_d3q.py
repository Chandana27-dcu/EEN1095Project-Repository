from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.actions_common import ACTIONS, get_rb_allocation
from src.config_d3q import CONFIG
from src.traffic_common import generate_traffic


class NetworkSlicingD3QEnv(gym.Env):
    """
    D3QN environment for dynamic network-slice resource allocation.

    Network slices:
        - eMBB
        - URLLC1
        - URLLC2
        - BE1

    Action space:
        Discrete common resource-allocation action set.

        D3QN selects one action index from the same common
        action set that will also be used by PPO.

    State values per slice:
        1. Throughput satisfaction
        2. Latency
        3. Packet Loss Ratio (PLR)
        4. Queue occupancy
        5. Channel condition
        6. Current traffic load

    Total state size:
        4 slices x 6 state features = 24

    Common reward:
        40% throughput satisfaction
        35% latency satisfaction
        25% PLR satisfaction

    Traffic:
        Low, Medium or High traffic is selected through
        CONFIG["LOAD_SCENARIO"].
    """

    metadata = {
        "render_modes": []
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self) -> None:
        super().__init__()

        # -----------------------------------------------------
        # Network slices
        # -----------------------------------------------------

        self.slices = list(
            CONFIG["SLICES"]
        )

        self.number_of_slices = len(
            self.slices
        )

        # -----------------------------------------------------
        # Validate common action space
        # -----------------------------------------------------

        if len(ACTIONS) != int(
            CONFIG["NUMBER_OF_ACTIONS"]
        ):
            raise ValueError(
                "Generated common action count does not "
                "match CONFIG['NUMBER_OF_ACTIONS']."
            )

        # -----------------------------------------------------
        # Network configuration
        # -----------------------------------------------------

        self.total_rb = int(
            CONFIG["TOTAL_RB"]
        )

        self.bits_per_rb = float(
            CONFIG["BITS_PER_RB"]
        )

        self.max_time = int(
            CONFIG["MAX_TIME"]
        )

        self.slot_duration_ms = float(
            CONFIG["SLOT_DURATION_MS"]
        )

        self.buffer_size = int(
            CONFIG["BUFFER_SIZE"]
        )

        self.queue_reference = float(
            CONFIG["QUEUE_REFERENCE"]
        )

        # -----------------------------------------------------
        # Minimum allocation
        # -----------------------------------------------------

        self.minimum_share = (
            float(
                CONFIG[
                    "MINIMUM_SHARE_PERCENT"
                ]
            )
            / 100.0
        )

        # -----------------------------------------------------
        # Traffic configuration
        # -----------------------------------------------------

        self.traffic_load = str(
            CONFIG["LOAD_SCENARIO"]
        )

        if self.traffic_load not in CONFIG[
            "TRAFFIC_SCENARIOS"
        ]:
            raise ValueError(
                f"Invalid traffic scenario: "
                f"{self.traffic_load}"
            )

        self.traffic_config = CONFIG[
            "TRAFFIC_SCENARIOS"
        ][self.traffic_load].copy()

        # -----------------------------------------------------
        # QoS and packet deadlines
        # -----------------------------------------------------

        self.qos = CONFIG["QOS"]

        self.deadline_ms = CONFIG[
            "DEADLINE_MS"
        ]

        # -----------------------------------------------------
        # Metric history windows
        # -----------------------------------------------------

        self.throughput_window = int(
            CONFIG["THROUGHPUT_WINDOW"]
        )

        self.latency_window = int(
            CONFIG["LATENCY_WINDOW"]
        )

        # -----------------------------------------------------
        # Reward configuration
        # -----------------------------------------------------

        self.reward_weights = CONFIG[
            "REWARD_WEIGHTS"
        ]

        # -----------------------------------------------------
        # Channel configuration
        # -----------------------------------------------------

        channel_config = CONFIG[
            "CHANNEL"
        ]

        self.channel_min = float(
            channel_config["min"]
        )

        self.channel_max = float(
            channel_config["max"]
        )

        self.channel_correlation = float(
            channel_config["correlation"]
        )

        self.channel_mean = float(
            channel_config["mean"]
        )

        self.channel_noise_std = float(
            channel_config["noise_std"]
        )

        # =====================================================
        # ACTION SPACE
        # =====================================================

        self.action_space = spaces.Discrete(
            len(ACTIONS)
        )

        # =====================================================
        # OBSERVATION SPACE
        # =====================================================

        expected_state_size = (
            self.number_of_slices * 6
        )

        if int(CONFIG["STATE_SIZE"]) != (
            expected_state_size
        ):
            raise ValueError(
                "CONFIG['STATE_SIZE'] does not match "
                "4 slices x 6 state features."
            )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(expected_state_size,),
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # Internal simulation variables
        # -----------------------------------------------------

        self.time = 0

        self.queue: dict[
            str,
            list[dict[str, Any]]
        ] = {}

        self.metrics: dict[
            str,
            dict[str, Any]
        ] = {}

        # Initial equal allocation representation.
        self.last_action = np.full(
            self.number_of_slices,
            1.0 / self.number_of_slices,
            dtype=np.float32,
        )

        self.reset()

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        """
        Reset the network environment.

        Using the same seed for D3QN and PPO during final
        evaluation helps ensure equivalent random traffic
        and channel realizations.
        """

        super().reset(seed=seed)

        del options

        self.time = 0

        # -----------------------------------------------------
        # Empty packet queues
        # -----------------------------------------------------

        self.queue = {
            slice_name: []
            for slice_name in self.slices
        }

        # -----------------------------------------------------
        # Initialize metrics
        # -----------------------------------------------------

        self.metrics = {

            # Moving-average throughput satisfaction.
            "throughput_percent": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Throughput percentage at every simulation step.
            "step_throughput_percent": {
                slice_name: []
                for slice_name in self.slices
            },

            # Offered bits during current simulation step.
            "step_offered_bits": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Successfully transmitted bits.
            "throughput_bits": {
                slice_name: 0.0
                for slice_name in self.slices
            },

            # Successful packet latency values.
            "latency_ms": {
                slice_name: []
                for slice_name in self.slices
            },

            # Packet loss ratio.
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
                slice_name:
                    self.channel_mean
                for slice_name in self.slices
            },

            # Number of new packets during current step.
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

        info = {
            "traffic_load":
                self.traffic_load,
        }

        return observation, info

    # =========================================================
    # STEP
    # =========================================================

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
        """
        Execute one simulation step.
        """

        self.time += 1

        # -----------------------------------------------------
        # Validate action
        # -----------------------------------------------------

        action_index = int(action)

        if not self.action_space.contains(
            action_index
        ):
            raise ValueError(
                f"Invalid D3QN action index: "
                f"{action_index}"
            )

        # -----------------------------------------------------
        # Convert selected action into allocation percentages
        # -----------------------------------------------------

        selected_action = ACTIONS[
            action_index
        ]

        allocation_shares = np.asarray(
            [
                selected_action[
                    slice_name
                ] / 100.0
                for slice_name
                in self.slices
            ],
            dtype=np.float32,
        )

        self.last_action = (
            allocation_shares.copy()
        )

        allocation_percent = {
            slice_name: float(
                selected_action[
                    slice_name
                ]
            )
            for slice_name
            in self.slices
        }

        # -----------------------------------------------------
        # Convert percentage allocation into RBs
        # -----------------------------------------------------

        rb_allocation = (
            get_rb_allocation(
                action_index=action_index,
                total_rbs=self.total_rb,
            )
        )

        # -----------------------------------------------------
        # Update time-varying channel
        # -----------------------------------------------------

        self._update_channel_conditions()

        # -----------------------------------------------------
        # Generate common traffic
        # -----------------------------------------------------

        generated_traffic = (
            generate_traffic(
                time_step=self.time,
                traffic_config=
                    self.traffic_config,
                deadline_config=
                    self.deadline_ms,
                slice_names=
                    self.slices,
                rng=self.np_random,
            )
        )

        # -----------------------------------------------------
        # Add generated packets to queues
        # -----------------------------------------------------

        for slice_name in self.slices:

            new_packets = (
                generated_traffic[
                    slice_name
                ]
            )

            offered_bits = sum(
                float(packet["size"])
                for packet
                in new_packets
            )

            self.metrics[
                "step_offered_bits"
            ][slice_name] = offered_bits

            self.metrics[
                "recent_arrivals"
            ][slice_name] = len(
                new_packets
            )

            for packet in new_packets:

                self.queue[
                    slice_name
                ].append(packet)

                self.metrics[
                    "arrivals"
                ][slice_name] += 1

        # -----------------------------------------------------
        # Record transmitted bits for current step
        # -----------------------------------------------------

        step_throughput_bits = {
            slice_name: 0.0
            for slice_name
            in self.slices
        }

        # -----------------------------------------------------
        # Serve all queues
        # -----------------------------------------------------

        for slice_name in self.slices:

            self._serve_slice_packets(
                slice_name=
                    slice_name,
                allocated_rbs=
                    rb_allocation[
                        slice_name
                    ],
                step_throughput_bits=
                    step_throughput_bits,
            )

        # -----------------------------------------------------
        # Update metrics
        # -----------------------------------------------------

        self._update_metrics(
            step_throughput_bits
        )

        # -----------------------------------------------------
        # Common reward calculation
        # -----------------------------------------------------

        reward = (
            self._calculate_reward(
                allocation_shares
            )
        )

        # -----------------------------------------------------
        # Episode termination
        # -----------------------------------------------------

        terminated = (
            self.time
            >= self.max_time
        )

        truncated = False

        observation = self.get_state()

        # -----------------------------------------------------
        # Additional evaluation information
        # -----------------------------------------------------

        info = {
            "traffic_load":
                self.traffic_load,

            "action_index":
                action_index,

            "allocation_percent":
                allocation_percent,

            "rb_allocation":
                rb_allocation,

            "metrics":
                self.get_episode_metrics(),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # =========================================================
    # PACKET SERVICE
    # =========================================================

    def _serve_slice_packets(
        self,
        slice_name: str,
        allocated_rbs: int | float,
        step_throughput_bits:
            dict[str, float],
    ) -> None:
        """
        Serve packets using FIFO scheduling.

        Transmission capacity depends on:
            allocated RBs
            x bits per RB
            x current channel factor
        """

        channel_factor = float(
            self.metrics[
                "channel"
            ][slice_name]
        )

        capacity_bits = (
            float(allocated_rbs)
            * self.bits_per_rb
            * channel_factor
        )

        remaining_queue: list[
            dict[str, Any]
        ] = []

        # -----------------------------------------------------
        # FIFO packet transmission
        # -----------------------------------------------------

        for packet in self.queue[
            slice_name
        ]:

            packet_size = float(
                packet["size"]
            )

            if packet_size <= capacity_bits:

                waiting_slots = max(
                    1,
                    self.time
                    - int(
                        packet["arrival"]
                    ),
                )

                latency_ms = (
                    waiting_slots
                    * self.slot_duration_ms
                )

                self.metrics[
                    "latency_ms"
                ][slice_name].append(
                    float(latency_ms)
                )

                self.metrics[
                    "throughput_bits"
                ][slice_name] += (
                    packet_size
                )

                step_throughput_bits[
                    slice_name
                ] += packet_size

                capacity_bits -= (
                    packet_size
                )

            else:
                remaining_queue.append(
                    packet
                )

        # -----------------------------------------------------
        # Drop expired packets
        # -----------------------------------------------------

        valid_packets: list[
            dict[str, Any]
        ] = []

        deadline_ms = float(
            self.deadline_ms[
                slice_name
            ]
        )

        for packet in remaining_queue:

            waiting_slots = max(
                1,
                self.time
                - int(
                    packet["arrival"]
                ),
            )

            waiting_time_ms = (
                waiting_slots
                * self.slot_duration_ms
            )

            if (
                waiting_time_ms
                > deadline_ms
            ):
                self.metrics[
                    "dropped"
                ][slice_name] += 1

            else:
                valid_packets.append(
                    packet
                )

        remaining_queue = (
            valid_packets
        )

        # -----------------------------------------------------
        # Apply queue buffer limit
        # -----------------------------------------------------

        if (
            len(remaining_queue)
            > self.buffer_size
        ):

            overflow_count = (
                len(remaining_queue)
                - self.buffer_size
            )

            self.metrics[
                "dropped"
            ][slice_name] += (
                overflow_count
            )

            # Keep oldest packets for FIFO behavior.
            remaining_queue = (
                remaining_queue[
                    :self.buffer_size
                ]
            )

        self.queue[
            slice_name
        ] = remaining_queue

    # =========================================================
    # CHANNEL MODEL
    # =========================================================

    def _update_channel_conditions(
        self,
    ) -> None:
        """
        Update channel quality using a correlated
        random process.

        This produces gradually varying radio
        conditions rather than independently
        random channel values at every step.
        """

        for slice_name in self.slices:

            previous_channel = float(
                self.metrics[
                    "channel"
                ][slice_name]
            )

            channel_noise = float(
                self.np_random.normal(
                    loc=0.0,
                    scale=
                        self.channel_noise_std,
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

            self.metrics[
                "channel"
            ][slice_name] = float(
                np.clip(
                    new_channel,
                    self.channel_min,
                    self.channel_max,
                )
            )

    # =========================================================
    # UPDATE PERFORMANCE METRICS
    # =========================================================

    def _update_metrics(
        self,
        step_throughput_bits:
            dict[str, float],
    ) -> None:
        """
        Update throughput and packet-loss metrics.
        """

        for slice_name in self.slices:

            offered_bits = float(
                self.metrics[
                    "step_offered_bits"
                ][slice_name]
            )

            served_bits = float(
                step_throughput_bits[
                    slice_name
                ]
            )

            # -------------------------------------------------
            # Throughput satisfaction
            # -------------------------------------------------

            if offered_bits > 0.0:

                throughput_percent = (
                    served_bits
                    / offered_bits
                ) * 100.0

            else:

                throughput_percent = (
                    100.0
                    if len(
                        self.queue[
                            slice_name
                        ]
                    ) == 0
                    else 0.0
                )

            throughput_percent = float(
                np.clip(
                    throughput_percent,
                    0.0,
                    100.0,
                )
            )

            throughput_history = (
                self.metrics[
                    "step_throughput_percent"
                ][slice_name]
            )

            throughput_history.append(
                throughput_percent
            )

            recent_throughput = (
                throughput_history[
                    -self.throughput_window:
                ]
            )

            self.metrics[
                "throughput_percent"
            ][slice_name] = float(
                np.mean(
                    recent_throughput
                )
                if recent_throughput
                else 0.0
            )

            # -------------------------------------------------
            # Packet Loss Ratio
            # -------------------------------------------------

            arrivals = int(
                self.metrics[
                    "arrivals"
                ][slice_name]
            )

            dropped = int(
                self.metrics[
                    "dropped"
                ][slice_name]
            )

            if arrivals > 0:

                plr_percent = (
                    dropped
                    / arrivals
                ) * 100.0

            else:
                plr_percent = 0.0

            self.metrics[
                "plr_percent"
            ][slice_name] = float(
                np.clip(
                    plr_percent,
                    0.0,
                    100.0,
                )
            )

    # =========================================================
    # COMMON REWARD FUNCTION
    # =========================================================

    def _calculate_reward(
        self,
        action: np.ndarray,
    ) -> float:
        """
        Common reward used for both D3QN and PPO.

        Reward:
            40% Throughput satisfaction
            35% Latency satisfaction
            25% PLR satisfaction

        The selected RB allocation influences network
        performance, but the action itself is not
        directly rewarded or penalized.
        """

        del action

        throughput_weight = float(
            self.reward_weights[
                "throughput"
            ]
        )

        latency_weight = float(
            self.reward_weights[
                "latency"
            ]
        )

        plr_weight = float(
            self.reward_weights[
                "plr"
            ]
        )

        slice_rewards: list[
            float
        ] = []

        for slice_name in self.slices:

            # -------------------------------------------------
            # Throughput score
            # -------------------------------------------------

            throughput_percent = float(
                self.metrics[
                    "throughput_percent"
                ][slice_name]
            )

            throughput_score = float(
                np.clip(
                    throughput_percent
                    / 100.0,
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # Latency score
            # -------------------------------------------------

            latency_history = (
                self.metrics[
                    "latency_ms"
                ][slice_name]
            )

            recent_latencies = (
                latency_history[
                    -self.latency_window:
                ]
            )

            if recent_latencies:

                average_latency_ms = (
                    float(
                        np.mean(
                            recent_latencies
                        )
                    )
                )

            else:

                # No successful packet transmission.
                # Do not automatically assign perfect latency.
                average_latency_ms = float(
                    self.qos[
                        slice_name
                    ][
                        "latency_req_ms"
                    ]
                )

            required_latency_ms = float(
                self.qos[
                    slice_name
                ][
                    "latency_req_ms"
                ]
            )

            latency_score = float(
                np.clip(
                    1.0
                    - (
                        average_latency_ms
                        / max(
                            required_latency_ms,
                            1e-8,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # PLR score
            # -------------------------------------------------

            plr_percent = float(
                self.metrics[
                    "plr_percent"
                ][slice_name]
            )

            required_plr = float(
                self.qos[
                    slice_name
                ][
                    "plr_req"
                ]
            )

            plr_score = float(
                np.clip(
                    1.0
                    - (
                        plr_percent
                        / max(
                            required_plr,
                            1e-8,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # Combined slice reward
            # -------------------------------------------------

            slice_reward = (
                throughput_weight
                * throughput_score

                + latency_weight
                * latency_score

                + plr_weight
                * plr_score
            )

            slice_rewards.append(
                float(slice_reward)
            )

        total_reward = float(
            np.mean(
                slice_rewards
            )
        )

        return float(
            np.clip(
                total_reward,
                0.0,
                1.0,
            )
        )

    # =========================================================
    # STATE REPRESENTATION
    # =========================================================

    def get_state(
        self,
    ) -> np.ndarray:
        """
        Return the normalized 24-dimensional state.

        For each slice:
            1. Throughput
            2. Latency
            3. PLR
            4. Queue occupancy
            5. Channel condition
            6. Traffic load
        """

        state: list[
            float
        ] = []

        for slice_name in self.slices:

            # -------------------------------------------------
            # Throughput
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Latency
            # -------------------------------------------------

            latency_history = (
                self.metrics[
                    "latency_ms"
                ][slice_name]
            )

            recent_latencies = (
                latency_history[
                    -self.latency_window:
                ]
            )

            average_latency_ms = float(
                np.mean(
                    recent_latencies
                )
                if recent_latencies
                else 0.0
            )

            required_latency_ms = float(
                self.qos[
                    slice_name
                ][
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

            # -------------------------------------------------
            # PLR
            # -------------------------------------------------

            plr_normalized = float(
                np.clip(
                    self.metrics[
                        "plr_percent"
                    ][slice_name]
                    / 100.0,
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # Queue occupancy
            # -------------------------------------------------

            queue_normalized = float(
                np.clip(
                    len(
                        self.queue[
                            slice_name
                        ]
                    )
                    / max(
                        self.queue_reference,
                        1.0,
                    ),
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # Channel condition
            # -------------------------------------------------

            channel_value = float(
                self.metrics[
                    "channel"
                ][slice_name]
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

            # -------------------------------------------------
            # Current traffic load
            # -------------------------------------------------

            recent_arrivals = int(
                self.metrics[
                    "recent_arrivals"
                ][slice_name]
            )

            traffic_load_normalized = float(
                np.clip(
                    recent_arrivals
                    / 20.0,
                    0.0,
                    1.0,
                )
            )

            # -------------------------------------------------
            # Append six features
            # -------------------------------------------------

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

        if (
            state_array.shape[0]
            != CONFIG["STATE_SIZE"]
        ):
            raise RuntimeError(
                f"State contains "
                f"{state_array.shape[0]} values, "
                f"but CONFIG['STATE_SIZE'] is "
                f"{CONFIG['STATE_SIZE']}."
            )

        return np.clip(
            state_array,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # EPISODE METRICS
    # =========================================================

    def get_episode_metrics(
        self,
    ) -> dict[
        str,
        dict[
            str,
            float | int
        ],
    ]:
        """
        Return readable evaluation metrics for all slices.
        """

        results: dict[
            str,
            dict[
                str,
                float | int
            ],
        ] = {}

        for slice_name in self.slices:

            # -------------------------------------------------
            # Latency and jitter
            # -------------------------------------------------

            latency_history = (
                self.metrics[
                    "latency_ms"
                ][slice_name]
            )

            if latency_history:

                average_latency_ms = float(
                    np.mean(
                        latency_history
                    )
                )

                jitter_ms = float(
                    np.std(
                        latency_history
                    )
                    if len(
                        latency_history
                    ) > 1
                    else 0.0
                )

            else:

                average_latency_ms = 0.0

                jitter_ms = 0.0

            # -------------------------------------------------
            # Average throughput
            # -------------------------------------------------

            throughput_history = (
                self.metrics[
                    "step_throughput_percent"
                ][slice_name]
            )

            average_throughput_percent = (
                float(
                    np.mean(
                        throughput_history
                    )
                )
                if throughput_history
                else 0.0
            )

            # -------------------------------------------------
            # Save slice metrics
            # -------------------------------------------------

            results[
                slice_name
            ] = {

                "average_throughput_percent":
                    average_throughput_percent,

                "current_throughput_percent":
                    float(
                        self.metrics[
                            "throughput_percent"
                        ][slice_name]
                    ),

                "average_latency_ms":
                    average_latency_ms,

                "jitter_ms":
                    jitter_ms,

                "plr_percent":
                    float(
                        self.metrics[
                            "plr_percent"
                        ][slice_name]
                    ),

                "queue_length":
                    int(
                        len(
                            self.queue[
                                slice_name
                            ]
                        )
                    ),

                "channel_condition":
                    float(
                        self.metrics[
                            "channel"
                        ][slice_name]
                    ),

                "arrivals":
                    int(
                        self.metrics[
                            "arrivals"
                        ][slice_name]
                    ),

                "dropped":
                    int(
                        self.metrics[
                            "dropped"
                        ][slice_name]
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