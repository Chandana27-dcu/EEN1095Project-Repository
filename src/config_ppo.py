"""
PPO configuration for dynamic network slicing.

IMPORTANT:
D3QN and PPO must use the same:
- network capacity
- traffic scenarios
- packet sizes
- state space
- action space
- QoS requirements
- packet deadlines
- channel conditions
- metric windows
- reward weights
- evaluation settings

Only algorithm-specific training hyperparameters
are allowed to differ.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# SELECT TRAFFIC SCENARIO
# ============================================================

# Supported:
# "low"
# "medium"
# "high"

LOAD_SCENARIO = "medium"


# ============================================================
# TRAFFIC SCENARIOS
#
# Packet sizes are represented in BITS.
#
# These values MUST be identical for D3QN and PPO.
# ============================================================

TRAFFIC_SCENARIOS: dict[str, dict[str, int]] = {

    "low": {
        "eMBB_LAMBDA": 3,
        "eMBB_PKT": 2000 * 8,

        "URLLC1_LAMBDA": 2,
        "URLLC1_PKT": 300 * 8,

        "URLLC2_PERIOD": 2,
        "URLLC2_PKT": 300 * 8,

        "BE1_LAMBDA": 1,
        "BE1_PKT": 1500 * 8,
    },

    "medium": {
        "eMBB_LAMBDA": 6,
        "eMBB_PKT": 2000 * 8,

        "URLLC1_LAMBDA": 4,
        "URLLC1_PKT": 300 * 8,

        "URLLC2_PERIOD": 1,
        "URLLC2_PKT": 300 * 8,

        "BE1_LAMBDA": 3,
        "BE1_PKT": 1500 * 8,
    },

    "high": {
        "eMBB_LAMBDA": 10,
        "eMBB_PKT": 2000 * 8,

        "URLLC1_LAMBDA": 7,
        "URLLC1_PKT": 300 * 8,

        "URLLC2_PERIOD": 1,
        "URLLC2_PKT": 300 * 8,

        "BE1_LAMBDA": 6,
        "BE1_PKT": 1500 * 8,
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

    "DEFAULT_TRAFFIC_LOAD": "medium",

    "SEED": 42,

    # --------------------------------------------------------
    # Network slices
    # --------------------------------------------------------

    "SLICES": [
        "eMBB",
        "URLLC1",
        "URLLC2",
        "BE1",
    ],

    # --------------------------------------------------------
    # Network capacity
    # --------------------------------------------------------

    "TOTAL_RB": 100,

    "BITS_PER_RB": 300,

    "MAX_TIME": 500,

    "SLOT_DURATION_MS": 1.0,

    "BUFFER_SIZE": 200,

    "QUEUE_REFERENCE": 200,

    # --------------------------------------------------------
    # COMMON ACTION SPACE
    #
    # PPO must choose from exactly the same candidate
    # resource-allocation patterns as D3QN.
    # --------------------------------------------------------

    "NUMBER_OF_ACTIONS": 155,

    "ACTION_STEP_PERCENT": 5,

    "MINIMUM_SHARE_PERCENT": 5,

    # --------------------------------------------------------
    # COMMON STATE SPACE
    # --------------------------------------------------------

    # Four slices × six state variables:
    #
    # 1. throughput
    # 2. latency
    # 3. PLR
    # 4. queue occupancy
    # 5. channel quality
    # 6. traffic load
    #
    # 4 × 6 = 24
    "STATE_SIZE": 24,

    # --------------------------------------------------------
    # Traffic configuration
    # --------------------------------------------------------

    "TRAFFIC_SCENARIOS": TRAFFIC_SCENARIOS,

    "TRAFFIC": TRAFFIC_SCENARIOS[
        LOAD_SCENARIO
    ].copy(),

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

    "DEADLINE_MS": {
        "eMBB": 150.0,
        "URLLC1": 20.0,
        "URLLC2": 10.0,
        "BE1": 300.0,
    },

    # --------------------------------------------------------
    # Dynamic channel model
    # --------------------------------------------------------

    "CHANNEL": {
        "min": 0.4,
        "max": 1.5,
        "correlation": 0.85,
        "mean": 1.0,
        "noise_std": 0.12,
    },

    # --------------------------------------------------------
    # Metric history windows
    # --------------------------------------------------------

    "THROUGHPUT_WINDOW": 20,

    "LATENCY_WINDOW": 100,

    # --------------------------------------------------------
    # Common reward weights
    # --------------------------------------------------------

    "REWARD_WEIGHTS": {
        "throughput": 0.40,
        "latency": 0.35,
        "plr": 0.25,
    },

    # ========================================================
    # PPO-SPECIFIC SETTINGS
    # ========================================================

    "PPO": {

        "learning_rate": 3e-4,

        "n_steps": 2048,

        "batch_size": 128,

        "gamma": 0.99,

        "hidden_neurons": 128,

        "total_timesteps": 100000,

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
        f"results/ppo_{LOAD_SCENARIO}"
        "_load_10_runs.csv"
    ),
}


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config() -> None:
    """Validate PPO and common experiment settings."""

    valid_scenarios = {
        "low",
        "medium",
        "high",
    }

    # --------------------------------------------------------
    # Traffic scenario
    # --------------------------------------------------------

    if CONFIG["LOAD_SCENARIO"] not in valid_scenarios:
        raise ValueError(
            "LOAD_SCENARIO must be "
            "'low', 'medium', or 'high'."
        )

    # --------------------------------------------------------
    # Slices
    # --------------------------------------------------------

    if len(CONFIG["SLICES"]) != 4:
        raise ValueError(
            "The experiment must contain four slices."
        )

    # --------------------------------------------------------
    # State space
    # --------------------------------------------------------

    expected_state_size = (
        len(CONFIG["SLICES"]) * 6
    )

    if CONFIG["STATE_SIZE"] != expected_state_size:
        raise ValueError(
            "STATE_SIZE must equal "
            "number of slices × 6."
        )

    # --------------------------------------------------------
    # Network parameters
    # --------------------------------------------------------

    if CONFIG["TOTAL_RB"] <= 0:
        raise ValueError(
            "TOTAL_RB must be greater than zero."
        )

    if CONFIG["BITS_PER_RB"] <= 0:
        raise ValueError(
            "BITS_PER_RB must be greater than zero."
        )

    if CONFIG["MAX_TIME"] <= 0:
        raise ValueError(
            "MAX_TIME must be greater than zero."
        )

    if CONFIG["BUFFER_SIZE"] <= 0:
        raise ValueError(
            "BUFFER_SIZE must be greater than zero."
        )

    # --------------------------------------------------------
    # Action space
    # --------------------------------------------------------

    if CONFIG["NUMBER_OF_ACTIONS"] <= 0:
        raise ValueError(
            "NUMBER_OF_ACTIONS must be positive."
        )

    if CONFIG["ACTION_STEP_PERCENT"] <= 0:
        raise ValueError(
            "ACTION_STEP_PERCENT must be positive."
        )

    minimum_share = CONFIG[
        "MINIMUM_SHARE_PERCENT"
    ]

    if not 0 < minimum_share < 100:
        raise ValueError(
            "MINIMUM_SHARE_PERCENT must be "
            "between 0 and 100."
        )

    minimum_total = (
        minimum_share
        * len(CONFIG["SLICES"])
    )

    if minimum_total >= 100:
        raise ValueError(
            "Minimum resource shares leave no "
            "resources available for allocation."
        )

    # --------------------------------------------------------
    # Reward weights
    # --------------------------------------------------------

    reward_total = sum(
        CONFIG["REWARD_WEIGHTS"].values()
    )

    if abs(reward_total - 1.0) > 1e-9:
        raise ValueError(
            "REWARD_WEIGHTS must add up to 1.0. "
            f"Current sum: {reward_total}"
        )

    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    channel = CONFIG["CHANNEL"]

    if channel["min"] <= 0:
        raise ValueError(
            "CHANNEL['min'] must be positive."
        )

    if channel["max"] <= channel["min"]:
        raise ValueError(
            "CHANNEL['max'] must be greater "
            "than CHANNEL['min']."
        )

    if not 0.0 <= channel["correlation"] <= 1.0:
        raise ValueError(
            "CHANNEL['correlation'] must be "
            "between 0 and 1."
        )

    # --------------------------------------------------------
    # PPO hyperparameters
    # --------------------------------------------------------

    ppo = CONFIG["PPO"]

    if ppo["learning_rate"] <= 0:
        raise ValueError(
            "PPO learning_rate must be positive."
        )

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
# DISPLAY CONFIGURATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PPO CONFIGURATION")
    print("=" * 60)

    print(
        f"Algorithm          : "
        f"{CONFIG['ALGORITHM']}"
    )

    print(
        f"Load scenario      : "
        f"{CONFIG['LOAD_SCENARIO']}"
    )

    print(
        f"Traffic            : "
        f"{CONFIG['TRAFFIC']}"
    )

    print(
        f"Total RBs          : "
        f"{CONFIG['TOTAL_RB']}"
    )

    print(
        f"Bits per RB        : "
        f"{CONFIG['BITS_PER_RB']}"
    )

    print(
        f"Maximum time       : "
        f"{CONFIG['MAX_TIME']}"
    )

    print(
        f"Buffer size        : "
        f"{CONFIG['BUFFER_SIZE']}"
    )

    print(
        f"State size         : "
        f"{CONFIG['STATE_SIZE']}"
    )

    print(
        f"Number of actions  : "
        f"{CONFIG['NUMBER_OF_ACTIONS']}"
    )

    print(
        f"Action step        : "
        f"{CONFIG['ACTION_STEP_PERCENT']}%"
    )

    print(
        f"Minimum share      : "
        f"{CONFIG['MINIMUM_SHARE_PERCENT']}%"
    )

    print(
        f"Hidden neurons     : "
        f"{CONFIG['PPO']['hidden_neurons']}"
    )

    print(
        f"Training steps     : "
        f"{CONFIG['PPO']['total_timesteps']}"
    )

    print(
        f"Model path         : "
        f"{CONFIG['MODEL_PATH']}"
    )

    print(
        f"Results path       : "
        f"{CONFIG['RESULT_PATH']}"
    )

    print("=" * 60)

    print(
        "Configuration validation passed."
    )