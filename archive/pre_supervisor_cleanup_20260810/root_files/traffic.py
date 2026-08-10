import numpy as np
from src.config import CONFIG


def generate_traffic(time_step):
    traffic = {"eMBB": [], "URLLC1": [], "URLLC2": [], "BE1": []}

    # eMBB
    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["eMBB_LAMBDA"])):
        traffic["eMBB"].append({"size": CONFIG["TRAFFIC"]["eMBB_PKT"], "arrival": time_step})

    # URLLC1
    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["URLLC1_LAMBDA"])):
        traffic["URLLC1"].append({"size": CONFIG["TRAFFIC"]["URLLC1_PKT"], "arrival": time_step})

    # URLLC2 periodic
    if time_step % CONFIG["TRAFFIC"]["URLLC2_PERIOD"] == 0:
        traffic["URLLC2"].append({"size": CONFIG["TRAFFIC"]["URLLC2_PKT"], "arrival": time_step})

    # BE1
    for _ in range(np.random.poisson(CONFIG["TRAFFIC"]["BE1_LAMBDA"])):
        traffic["BE1"].append({"size": CONFIG["TRAFFIC"]["BE1_PKT"], "arrival": time_step})

    return traffic
