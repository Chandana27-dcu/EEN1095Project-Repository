CONFIG = {
    # ======================================================
    # Network parameters
    # ======================================================

    "TOTAL_RB": 100,
    "BITS_PER_RB": 300,

    # Number of simulation steps in one episode.
    "MAX_TIME": 500,

    # One simulation step represents 1 millisecond.
    "SLOT_DURATION_MS": 1.0,

    # Maximum of 200 queued packets/transactions per slice.
    "BUFFER_SIZE": 200,

    # Reference value used to normalize queue occupancy.
    "QUEUE_REFERENCE": 200,

    # ======================================================
    # Network slices
    # ======================================================

    "SLICES": [
        "eMBB",
        "URLLC1",
        "URLLC2",
        "BE1",
    ],

    # ======================================================
    # D3QN action space
    # ======================================================

    # D3QN selects one of 155 predefined allocation settings.
    "NUMBER_OF_ACTIONS": 155,

    # Resource allocations are generated in 5% increments.
    "ACTION_STEP_PERCENT": 5,

    # Each slice receives at least 5%.
    "MINIMUM_SHARE_PERCENT": 5,

    # ======================================================
    # Traffic model — medium load
    #
    # Packet sizes are represented in bits.
    # ======================================================

    "TRAFFIC": {
        # eMBB/video traffic
        "eMBB_LAMBDA": 3,
        "eMBB_PKT": 2000,

        # URLLC1/voice traffic
        "URLLC1_LAMBDA": 2,
        "URLLC1_PKT": 300,

        # URLLC2/gaming traffic
        "URLLC2_PERIOD": 2,
        "URLLC2_PKT": 150,

        # Best-effort traffic
        "BE1_LAMBDA": 1,
        "BE1_PKT": 500,
    },

    # ======================================================
    # QoS requirements
    #
    # Throughput requirement: bits per simulation step
    # Latency requirement: milliseconds
    # PLR requirement: percentage
    # ======================================================

    "QOS": {
        "eMBB": {
            "throughput_req": 12000,
            "latency_req_ms": 100.0,
            "plr_req": 5.0,
        },

        "URLLC1": {
            "throughput_req": 1200,
            "latency_req_ms": 20.0,
            "plr_req": 1.0,
        },

        "URLLC2": {
            "throughput_req": 150,
            "latency_req_ms": 10.0,
            "plr_req": 1.0,
        },

        "BE1": {
            "throughput_req": 1500,
            "latency_req_ms": 200.0,
            "plr_req": 10.0,
        },
    },

    # ======================================================
    # Packet deadlines
    #
    # Packets waiting longer than these values are dropped.
    # ======================================================

    "DEADLINE_MS": {
        "eMBB": 150.0,
        "URLLC1": 20.0,
        "URLLC2": 10.0,
        "BE1": 300.0,
    },

    # ======================================================
    # Dynamic channel model
    #
    # Time-correlated channel quality represents changing
    # radio/network conditions such as fading and interference.
    # ======================================================

    "CHANNEL": {
        "min": 0.4,
        "max": 1.5,
        "correlation": 0.85,
        "mean": 1.0,
        "noise_std": 0.12,
    },

    # ======================================================
    # Metric history windows
    # ======================================================

    "THROUGHPUT_WINDOW": 20,
    "LATENCY_WINDOW": 100,

    # ======================================================
    # Reward weights
    #
    # All reward metrics are converted to the range 0–1.
    # The weights also sum to exactly 1.0.
    #
    # Reward is calculated independently for all four slices
    # and then averaged.
    # ======================================================

    "REWARD_WEIGHTS": {
        "throughput": 0.40,
        "latency": 0.35,
        "plr": 0.25,
    },

    # ======================================================
    # Double Dueling DQN network
    # ======================================================

    # Four slices × six normalized state features.
    "STATE_SIZE": 24,

    # Two hidden layers with 128 neurons.
    "HIDDEN_SIZE": 128,

    "LEARNING_RATE": 3e-4,
    "GAMMA": 0.99,

    # ======================================================
    # Prioritized Experience Replay
    #
    # This is different from BUFFER_SIZE.
    # BUFFER_SIZE is the packet queue capacity per slice.
    # REPLAY_BUFFER_SIZE stores RL transitions.
    # ======================================================

    "REPLAY_BUFFER_SIZE": 100000,
    "BATCH_SIZE": 128,

    "PER_ALPHA": 0.6,
    "PER_BETA_START": 0.4,
    "PER_BETA_FRAMES": 100000,
    "PER_EPSILON": 1e-5,

    # ======================================================
    # D3QN training
    # ======================================================

    "EPISODES": 200,

    "EPS_START": 1.0,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.995,

    # Update the target Q-network every 10 episodes.
    "TARGET_UPDATE": 10,

    # Start learning after collecting 1000 transitions.
    "LEARNING_STARTS": 1000,

    # Perform one optimization step per environment step.
    "TRAIN_FREQUENCY": 1,

    # Gradient clipping.
    "MAX_GRAD_NORM": 10.0,

    # ======================================================
    # Evaluation settings
    # ======================================================

    "EVALUATION": {
        "number_of_runs": 10,
        "base_seed": 42,
    },
}