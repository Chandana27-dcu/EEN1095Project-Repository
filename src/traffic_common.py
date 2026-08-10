from __future__ import annotations

from typing import Any

import numpy as np


def generate_traffic(
    time_step: int,
    traffic_config: dict[str, Any],
    deadline_config: dict[str, float],
    slice_names: list[str] | tuple[str, ...],
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate common traffic for D3QN, PPO and baseline.

    Traffic models:
        eMBB   : Poisson arrivals
        URLLC1 : Poisson arrivals
        URLLC2 : Periodic arrivals
        BE1    : Poisson arrivals
    """

    if rng is None:
        rng = np.random.default_rng()

    if time_step < 0:
        raise ValueError(
            "time_step cannot be negative."
        )

    required_slices = {
        "eMBB",
        "URLLC1",
        "URLLC2",
        "BE1",
    }

    if set(slice_names) != required_slices:
        raise ValueError(
            "slice_names must contain exactly "
            "eMBB, URLLC1, URLLC2 and BE1."
        )

    required_parameters = {
        "eMBB_LAMBDA",
        "eMBB_PKT",
        "URLLC1_LAMBDA",
        "URLLC1_PKT",
        "URLLC2_PERIOD",
        "URLLC2_PKT",
        "BE1_LAMBDA",
        "BE1_PKT",
    }

    missing_parameters = (
        required_parameters
        - set(traffic_config.keys())
    )

    if missing_parameters:
        raise ValueError(
            "Missing traffic parameters: "
            f"{sorted(missing_parameters)}"
        )

    if traffic_config["eMBB_LAMBDA"] < 0:
        raise ValueError(
            "eMBB_LAMBDA cannot be negative."
        )

    if traffic_config["URLLC1_LAMBDA"] < 0:
        raise ValueError(
            "URLLC1_LAMBDA cannot be negative."
        )

    if traffic_config["BE1_LAMBDA"] < 0:
        raise ValueError(
            "BE1_LAMBDA cannot be negative."
        )

    if traffic_config["URLLC2_PERIOD"] <= 0:
        raise ValueError(
            "URLLC2_PERIOD must be greater than zero."
        )

    for key in (
        "eMBB_PKT",
        "URLLC1_PKT",
        "URLLC2_PKT",
        "BE1_PKT",
    ):
        if traffic_config[key] <= 0:
            raise ValueError(
                f"{key} must be greater than zero."
            )

    traffic: dict[
        str,
        list[dict[str, Any]],
    ] = {
        slice_name: []
        for slice_name in slice_names
    }

    # ======================================================
    # eMBB - Poisson arrivals
    # ======================================================

    embb_packets = int(
        rng.poisson(
            traffic_config[
                "eMBB_LAMBDA"
            ]
        )
    )

    for _ in range(embb_packets):
        traffic["eMBB"].append(
            {
                "size": int(
                    traffic_config[
                        "eMBB_PKT"
                    ]
                ),
                "arrival": int(
                    time_step
                ),
                "deadline_ms": float(
                    deadline_config[
                        "eMBB"
                    ]
                ),
            }
        )

    # ======================================================
    # URLLC1 - Poisson arrivals
    # ======================================================

    urllc1_packets = int(
        rng.poisson(
            traffic_config[
                "URLLC1_LAMBDA"
            ]
        )
    )

    for _ in range(
        urllc1_packets
    ):
        traffic["URLLC1"].append(
            {
                "size": int(
                    traffic_config[
                        "URLLC1_PKT"
                    ]
                ),
                "arrival": int(
                    time_step
                ),
                "deadline_ms": float(
                    deadline_config[
                        "URLLC1"
                    ]
                ),
            }
        )

    # ======================================================
    # URLLC2 - Periodic traffic
    # ======================================================

    urllc2_period = int(
        traffic_config[
            "URLLC2_PERIOD"
        ]
    )

    if (
        time_step
        % urllc2_period
        == 0
    ):
        traffic["URLLC2"].append(
            {
                "size": int(
                    traffic_config[
                        "URLLC2_PKT"
                    ]
                ),
                "arrival": int(
                    time_step
                ),
                "deadline_ms": float(
                    deadline_config[
                        "URLLC2"
                    ]
                ),
            }
        )

    # ======================================================
    # BE1 - Poisson arrivals
    # ======================================================

    be1_packets = int(
        rng.poisson(
            traffic_config[
                "BE1_LAMBDA"
            ]
        )
    )

    for _ in range(
        be1_packets
    ):
        traffic["BE1"].append(
            {
                "size": int(
                    traffic_config[
                        "BE1_PKT"
                    ]
                ),
                "arrival": int(
                    time_step
                ),
                "deadline_ms": float(
                    deadline_config[
                        "BE1"
                    ]
                ),
            }
        )

    return traffic