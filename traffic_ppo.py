import numpy as np
from src.config_ppo import CONFIG


def generate_traffic(time_step):
    traffic = {s: [] for s in CONFIG["SLICES"]}

    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["eMBB_LAMBDA"])):
        traffic["eMBB"].append({
            "size": CONFIG["TRAFFIC"]["eMBB_PKT"],
            "arrival": time_step
        })

    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["URLLC1_LAMBDA"])):
        traffic["URLLC1"].append({
            "size": CONFIG["TRAFFIC"]["URLLC1_PKT"],
            "arrival": time_step
        })

    if time_step % CONFIG["TRAFFIC"]["URLLC2_PERIOD"] == 0:
        traffic["URLLC2"].append({
            "size": CONFIG["TRAFFIC"]["URLLC2_PKT"],
            "arrival": time_step
        })

    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["BE1_LAMBDA"])):
        traffic["BE1"].append({
            "size": CONFIG["TRAFFIC"]["BE1_PKT"],
            "arrival": time_step
        })

    return traffic