from __future__ import annotations

import numpy as np

from src.config_d3q import CONFIG


SLICE_NAMES = CONFIG["SLICES"]


def generate_traffic(
    time_step: int,
    rng: np.random.Generator | None = None,
    load_multiplier: float = 1.0,
) -> dict:
    """
    Generate traffic for all network slices.

    Parameters
    ----------
    time_step : int
        Current simulation time step.

    rng : numpy.random.Generator
        Random number generator.

    load_multiplier : float
        Used to simulate low/medium/high traffic load.

    Returns
    -------
    Dictionary of packets for each slice.
    """

    if rng is None:
        rng = np.random.default_rng()

    traffic = {
        "eMBB": [],
        "URLLC1": [],
        "URLLC2": [],
        "BE1": [],
    }

    traffic_cfg = CONFIG["TRAFFIC"]
    deadline_cfg = CONFIG["DEADLINE_MS"]

    # ==========================================================
    # eMBB
    # ==========================================================

    embb_packets = rng.poisson(
        traffic_cfg["eMBB_LAMBDA"] * load_multiplier
    )

    for _ in range(embb_packets):

        traffic["eMBB"].append(
            {
                "size": traffic_cfg["eMBB_PKT"],
                "arrival": time_step,
                "deadline_ms": deadline_cfg["eMBB"],
            }
        )

    # ==========================================================
    # URLLC1
    # ==========================================================

    urllc1_packets = rng.poisson(
        traffic_cfg["URLLC1_LAMBDA"] * load_multiplier
    )

    for _ in range(urllc1_packets):

        traffic["URLLC1"].append(
            {
                "size": traffic_cfg["URLLC1_PKT"],
                "arrival": time_step,
                "deadline_ms": deadline_cfg["URLLC1"],
            }
        )

    # ==========================================================
    # URLLC2 (Periodic)
    # ==========================================================

    if (
        time_step
        % traffic_cfg["URLLC2_PERIOD"]
        == 0
    ):

        traffic["URLLC2"].append(
            {
                "size": traffic_cfg["URLLC2_PKT"],
                "arrival": time_step,
                "deadline_ms": deadline_cfg["URLLC2"],
            }
        )

    # ==========================================================
    # Best Effort
    # ==========================================================

    be_packets = rng.poisson(
        traffic_cfg["BE1_LAMBDA"] * load_multiplier
    )

    for _ in range(be_packets):

        traffic["BE1"].append(
            {
                "size": traffic_cfg["BE1_PKT"],
                "arrival": time_step,
                "deadline_ms": deadline_cfg["BE1"],
            }
        )

    return traffic


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    print("=" * 60)
    print("Traffic Generator Test")
    print("=" * 60)

    for t in range(1, 6):

        traffic = generate_traffic(
            time_step=t,
            rng=rng,
            load_multiplier=1.0,
        )

        print(f"\nTime Step {t}")

        for s in SLICE_NAMES:

            print(
                f"{s:<8}: "
                f"{len(traffic[s])} packets"
            )

            if len(traffic[s]) > 0:

                print(
                    " First packet:",
                    traffic[s][0],
                )