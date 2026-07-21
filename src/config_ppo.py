"""
PPO configuration for dynamic network slicing.

The PPO agent uses a continuous action space to allocate network resources
among four slices:

1. eMBB
2. URLLC1
3. URLLC2
4. BE1

Change LOAD_SCENARIO to "low", "medium", or "high" before training or
evaluating a particular traffic condition.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# SELECT THE TRAFFIC SCENARIO
# ============================================================

# Supported values:
# "low"
# "medium"
# "high"

LOAD_SCENARIO = "medium"


# ============================================================
# TRAFFIC SCENARIOS
# ============================================================

TRAFFIC_SCENARIOS: dict[str, dict[str, int]] = {
    "low": {
        "eMBB_LAMBDA": 3,
        "eMBB_PKT": 2000,

        "URLLC1_LAMBDA": 2,
        "URLLC1_PKT": 300,

        "URLLC2_PERIOD": 2,
        "URLLC2_PKT": 150,

        "BE1_LAMBDA": 1,
        "BE1_PKT": 500,
    },

    "medium": {
        "eMBB_LAMBDA": 6,
        "eMBB_PKT": 2000,

        "URLLC1_LAMBDA": 4,
        "URLLC1_PKT": 300,

        "URLLC2_PERIOD": 1,
        "URLLC2_PKT": 150,

        "BE1_LAMBDA": 3,
        "BE1_PKT": 500,
    },

    "high": {
        "eMBB_LAMBDA": 10,
        "eMBB_PKT": 2000,

        "URLLC1_LAMBDA": 7,
        "URLLC1_PKT": 300,

        "URLLC2_PERIOD": 1,
        "URLLC2_PKT": 150,

        "BE1_LAMBDA": 6,
        "BE1_PKT": 500,
    },
}


if LOAD_SCENARIO not in TRAFFIC_SCENARIOS:
    raise ValueError(
        f"Invalid LOAD_SCENARIO '{LOAD_SCENARIO}'. "
        "Choose 'low', 'medium', or 'high'."
    )


# ============================================================
# MAIN PPO CONFIGURATION
# ============================================================

CONFIG: dict[str, Any] = {

    # --------------------------------------------------------
    # General experiment settings
    # --------------------------------------------------------

    "ALGORITHM": "PPO",

    "LOAD_SCENARIO": LOAD_SCENARIO,

    "SEED": 42,

    # Four network slices used in the simulation.
    "SLICES": [
        "eMBB",
        "URLLC1",
        "URLLC2",
        "BE1",
    ],

    # --------------------------------------------------------
    # Network capacity
    # --------------------------------------------------------

    # Total available resource blocks.
    "TOTAL_RB": 100,

    # Number of useful bits that one RB can transmit per slot
    # under the nominal channel condition.
    "BITS_PER_RB": 300,

    # Duration of one episode in simulation steps.
    "MAX_TIME": 500,

    # Duration of one simulation time slot.
    "SLOT_DURATION_MS": 1.0,

    # Maximum number of packets stored in each slice queue.
    "BUFFER_SIZE": 200,

    # Used for state normalisation.
    "QUEUE_REFERENCE": 200,

    # Minimum resource share for each slice.
    #
    # The environment expects this as a fraction.
    # 0.05 means a minimum allocation of 5%.
    "MINIMUM_SHARE": 0.05,

    # --------------------------------------------------------
    # State and action dimensions
    # --------------------------------------------------------

    # Four slices × six state variables:
    #
    # throughput
    # latency
    # PLR
    # queue length
    # channel quality
    # traffic load
    "STATE_SIZE": 24,

    # PPO produces one continuous allocation value per slice.
    "ACTION_SIZE": 4,

    # --------------------------------------------------------
    # Traffic configuration
    # --------------------------------------------------------

    "TRAFFIC": TRAFFIC_SCENARIOS[LOAD_SCENARIO].copy(),

    # --------------------------------------------------------
    # QoS requirements
    # --------------------------------------------------------

    "QOS": {
        "eMBB": {
            "throughput_req": 12000.0,
            "latency_req_ms": 100.0,
            "plr_req": 5.0,
        },

        "URLLC1": {
            "throughput_req": 1200.0,
            "latency_req_ms": 20.0,
            "plr_req": 1.0,
        },

        "URLLC2": {
            "throughput_req": 150.0,
            "latency_req_ms": 10.0,
            "plr_req": 1.0,
        },

        "BE1": {
            "throughput_req": 1500.0,
            "latency_req_ms": 200.0,
            "plr_req": 10.0,
        },
    },

    # --------------------------------------------------------
    # Packet deadlines
    # --------------------------------------------------------

    # A packet exceeding its deadline can be treated as dropped
    # or expired by the environment.
    "DEADLINE_MS": {
        "eMBB": 150.0,
        "URLLC1": 20.0,
        "URLLC2": 10.0,
        "BE1": 300.0,
    },

    # --------------------------------------------------------
    # Time-varying channel configuration
    # --------------------------------------------------------

    "CHANNEL": {
        # Minimum channel-quality multiplier.
        "min": 0.4,

        # Maximum channel-quality multiplier.
        "max": 1.5,

        # Temporal correlation between consecutive channel states.
        "correlation": 0.85,

        # Average channel-quality multiplier.
        "mean": 1.0,

        # Standard deviation of channel variation.
        "noise_std": 0.12,
    },

    # --------------------------------------------------------
    # Metric windows
    # --------------------------------------------------------

    # Number of recent steps used for throughput calculation.
    "THROUGHPUT_WINDOW": 20,

    # Number of recent packet delays used for latency calculation.
    "LATENCY_WINDOW": 100,

    # --------------------------------------------------------
    # Reward weights
    # --------------------------------------------------------

    # These values must add up to 1.0.
    "REWARD_WEIGHTS": {
        "throughput": 0.40,
        "latency": 0.35,
        "plr": 0.25,
    },

    # --------------------------------------------------------
    # PPO hyperparameters
    # --------------------------------------------------------

    "PPO": {
        # Learning rate used by the optimizer.
        "learning_rate": 3e-4,

        # Number of environment steps collected before an update.
        "n_steps": 2048,

        # Mini-batch size used during PPO optimisation.
        "batch_size": 128,

        # Discount factor.
        "gamma": 0.99,

        # Number of neurons in each hidden layer.
        "hidden_neurons": 128,

        # Total PPO training duration.
        "total_timesteps": 100_000,

        # Additional standard PPO settings.
        "n_epochs": 10,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.01,
        "vf_coef": 0.50,
        "max_grad_norm": 0.50,
    },

    # --------------------------------------------------------
    # Evaluation settings
    # --------------------------------------------------------

    "EVALUATION": {
        "number_of_runs": 10,
        "base_seed": 42,
        "deterministic": True,
    },

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    "MODEL_PATH": (
        f"models/ppo_{LOAD_SCENARIO}_load.zip"
    ),

    "RESULT_PATH": (
        f"results/ppo_{LOAD_SCENARIO}_load_10_runs.csv"
    ),
}


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config() -> None:
    """Check the most important configuration values."""

    valid_scenarios = {"low", "medium", "high"}

    if CONFIG["LOAD_SCENARIO"] not in valid_scenarios:
        raise ValueError(
            "LOAD_SCENARIO must be 'low', 'medium', or 'high'."
        )

    if len(CONFIG["SLICES"]) != CONFIG["ACTION_SIZE"]:
        raise ValueError(
            "ACTION_SIZE must match the number of slices."
        )

    if CONFIG["STATE_SIZE"] != len(CONFIG["SLICES"]) * 6:
        raise ValueError(
            "STATE_SIZE must equal number of slices multiplied by 6."
        )

    if CONFIG["TOTAL_RB"] <= 0:
        raise ValueError("TOTAL_RB must be greater than zero.")

    if CONFIG["BITS_PER_RB"] <= 0:
        raise ValueError("BITS_PER_RB must be greater than zero.")

    if CONFIG["MAX_TIME"] <= 0:
        raise ValueError("MAX_TIME must be greater than zero.")

    if CONFIG["BUFFER_SIZE"] <= 0:
        raise ValueError("BUFFER_SIZE must be greater than zero.")

    if not 0.0 <= CONFIG["MINIMUM_SHARE"] < 1.0:
        raise ValueError(
            "MINIMUM_SHARE must be between 0 and 1."
        )

    minimum_total = (
        CONFIG["MINIMUM_SHARE"] * len(CONFIG["SLICES"])
    )

    if minimum_total >= 1.0:
        raise ValueError(
            "The total minimum resource allocation must be below 100%."
        )

    reward_weight_total = sum(
        CONFIG["REWARD_WEIGHTS"].values()
    )

    if abs(reward_weight_total - 1.0) > 1e-9:
        raise ValueError(
            "REWARD_WEIGHTS must add up to 1.0. "
            f"Current sum: {reward_weight_total}"
        )

    channel = CONFIG["CHANNEL"]

    if channel["min"] <= 0:
        raise ValueError(
            "CHANNEL['min'] must be greater than zero."
        )

    if channel["max"] <= channel["min"]:
        raise ValueError(
            "CHANNEL['max'] must be greater than CHANNEL['min']."
        )

    if not 0.0 <= channel["correlation"] <= 1.0:
        raise ValueError(
            "CHANNEL['correlation'] must be between 0 and 1."
        )

    ppo = CONFIG["PPO"]

    if ppo["batch_size"] > ppo["n_steps"]:
        raise ValueError(
            "PPO batch_size cannot exceed n_steps."
        )

    if ppo["hidden_neurons"] <= 0:
        raise ValueError(
            "PPO hidden_neurons must be positive."
        )

    if ppo["total_timesteps"] <= 0:
        raise ValueError(
            "PPO total_timesteps must be positive."
        )


validate_config()


# ============================================================
# DISPLAY CONFIGURATION WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PPO CONFIGURATION")
    print("=" * 60)

    print(f"Algorithm       : {CONFIG['ALGORITHM']}")
    print(f"Load scenario   : {CONFIG['LOAD_SCENARIO']}")
    print(f"Traffic         : {CONFIG['TRAFFIC']}")
    print(f"Total RBs       : {CONFIG['TOTAL_RB']}")
    print(f"Buffer size     : {CONFIG['BUFFER_SIZE']}")
    print(f"State size      : {CONFIG['STATE_SIZE']}")
    print(f"Action size     : {CONFIG['ACTION_SIZE']}")
    print(f"Hidden neurons  : {CONFIG['PPO']['hidden_neurons']}")
    print(f"Training steps  : {CONFIG['PPO']['total_timesteps']}")
    print(f"Model path      : {CONFIG['MODEL_PATH']}")
    print(f"Results path    : {CONFIG['RESULT_PATH']}")

    print("=" * 60)
    print("Configuration validation passed.")