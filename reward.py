import numpy as np
from src.config import CONFIG


def compute_reward(metrics):
    reward = 0

    # URLLC2
    if metrics["latency"]["URLLC2"]:
        reward += CONFIG["REWARD"]["URLLC2_LATENCY_WEIGHT"] * np.mean(metrics["latency"]["URLLC2"])
    reward += CONFIG["REWARD"]["URLLC2_PLR_WEIGHT"] * metrics["plr"]["URLLC2"]

    # URLLC1
    if metrics["latency"]["URLLC1"]:
        reward += CONFIG["REWARD"]["URLLC1_LATENCY_WEIGHT"] * np.mean(metrics["latency"]["URLLC1"])
    reward += CONFIG["REWARD"]["URLLC1_PLR_WEIGHT"] * metrics["plr"]["URLLC1"]

    # eMBB throughput
    reward += CONFIG["REWARD"]["EMBB_TP_WEIGHT"] * metrics["throughput"]["eMBB"]

    # BE1 throughput
    reward += CONFIG["REWARD"]["BE1_TP_WEIGHT"] * metrics["throughput"]["BE1"]

    # Global latency penalty
    all_lat = []
    for s in metrics["latency"]:
        all_lat.extend(metrics["latency"][s])
    if all_lat:
        reward += CONFIG["REWARD"]["GLOBAL_LAT_PENALTY"] * np.mean(all_lat)

    return reward
