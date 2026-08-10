import numpy as np
from config_d3q import CONFIG


def update_metrics(metrics, queue, time_step):
    # URLLC deadlines
    for slice_name, deadline in [("URLLC1", CONFIG["URLLC1_DEADLINE"]),
                                 ("URLLC2", CONFIG["URLLC2_DEADLINE"])]:

        lost = 0
        kept = []

        for pkt in queue[slice_name]:
            delay = time_step - pkt["arrival"]
            if delay > deadline:
                lost += 1
            else:
                kept.append(pkt)

        queue[slice_name] = kept

        arrivals = metrics["arrivals"][slice_name]
        metrics["plr"][slice_name] = lost / arrivals if arrivals > 0 else 0

    # eMBB and BE1 PLR = 0
    metrics["plr"]["eMBB"] = 0
    metrics["plr"]["BE1"] = 0
