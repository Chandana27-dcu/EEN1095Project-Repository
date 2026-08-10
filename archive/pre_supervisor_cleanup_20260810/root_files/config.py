CONFIG = {
    "TOTAL_RB": 50,
    "MAX_TIME": 200,
    "BITS_PER_RB": 180,

    "ACTIONS": [
        {"eMBB": 25, "URLLC1": 10, "URLLC2": 10, "BE1": 5},
        {"eMBB": 20, "URLLC1": 15, "URLLC2": 10, "BE1": 5},
        {"eMBB": 15, "URLLC1": 20, "URLLC2": 10, "BE1": 5},
        {"eMBB": 10, "URLLC1": 20, "URLLC2": 15, "BE1": 5},
        {"eMBB": 30, "URLLC1": 10, "URLLC2": 5, "BE1": 5}
    ],

    "TRAFFIC": {
        "eMBB_LAMBDA": 4,
        "eMBB_PKT": 2000,

        "URLLC1_LAMBDA": 3,
        "URLLC1_PKT": 300,

        "URLLC2_PERIOD": 2,
        "URLLC2_PKT": 100,

        "BE1_LAMBDA": 2,
        "BE1_PKT": 500
    },

    "URLLC1_DEADLINE": 5,
    "URLLC2_DEADLINE": 2,

    "REWARD": {
        "URLLC2_LATENCY_WEIGHT": -5,
        "URLLC2_PLR_WEIGHT": -10,

        "URLLC1_LATENCY_WEIGHT": -3,
        "URLLC1_PLR_WEIGHT": -6,

        "EMBB_TP_WEIGHT": 0.001,
        "BE1_TP_WEIGHT": 0.0005,

        "GLOBAL_LAT_PENALTY": -3.0
    },

    "LR": 0.0005,
    "GAMMA": 0.99,
    "BUFFER_SIZE": 50000,
    "BATCH_SIZE": 64,
    "EPISODES": 500,
    "EPS_START": 1.0,
    "EPS_END": 0.01,
    "EPS_DECAY": 0.99,
    "TARGET_UPDATE": 10
}
