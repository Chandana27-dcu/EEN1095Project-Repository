from __future__ import annotations

from itertools import product

import numpy as np

from src.config_d3q import CONFIG


# ============================================================
# COMMON RESOURCE-ALLOCATION ACTION SPACE
#
# This file is shared by D3QN and PPO so that both algorithms
# operate over exactly the same discrete action set.
# ============================================================

SLICE_NAMES = tuple(CONFIG["SLICES"])


# ============================================================
# GENERATE ALL FEASIBLE ACTIONS
# ============================================================

def generate_all_valid_actions(
    step_percent: int,
    minimum_percent: int,
) -> list[dict[str, int]]:
    """
    Generate every feasible percentage-based RB allocation.

    Conditions:
        1. Each slice allocation is a multiple of step_percent.
        2. Each slice receives at least minimum_percent.
        3. The allocations across all slices sum to exactly 100%.

    Example:
        {
            "eMBB": 40,
            "URLLC1": 25,
            "URLLC2": 20,
            "BE1": 15,
        }
    """

    if step_percent <= 0:
        raise ValueError(
            "ACTION_STEP_PERCENT must be greater than zero."
        )

    if 100 % step_percent != 0:
        raise ValueError(
            "ACTION_STEP_PERCENT must divide 100 exactly."
        )

    if minimum_percent < 0:
        raise ValueError(
            "MINIMUM_SHARE_PERCENT cannot be negative."
        )

    if minimum_percent % step_percent != 0:
        raise ValueError(
            "MINIMUM_SHARE_PERCENT must be a multiple of "
            "ACTION_STEP_PERCENT."
        )

    if minimum_percent * len(SLICE_NAMES) >= 100:
        raise ValueError(
            "The combined minimum allocation must be below 100%."
        )

    possible_values = range(
        minimum_percent,
        101,
        step_percent,
    )

    valid_actions: list[dict[str, int]] = []

    for allocations in product(
        possible_values,
        repeat=len(SLICE_NAMES),
    ):
        if sum(allocations) != 100:
            continue

        action = {
            slice_name: int(value)
            for slice_name, value in zip(
                SLICE_NAMES,
                allocations,
            )
        }

        valid_actions.append(action)

    # Deterministic order for reproducibility.
    valid_actions.sort(
        key=lambda action: tuple(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )
    )

    return valid_actions


# ============================================================
# ACTION CONVERSION
# ============================================================

def action_to_array(
    action: dict[str, int],
) -> np.ndarray:
    """
    Convert an allocation dictionary to a NumPy array in
    the fixed slice order:

        [eMBB, URLLC1, URLLC2, BE1]
    """

    return np.asarray(
        [
            action[slice_name]
            for slice_name in SLICE_NAMES
        ],
        dtype=np.float32,
    )


# ============================================================
# SELECT REPRESENTATIVE ACTIONS
# ============================================================

def select_representative_actions(
    all_actions: list[dict[str, int]],
    number_of_actions: int,
) -> list[dict[str, int]]:
    """
    Select a diverse subset of feasible allocations using
    deterministic farthest-point sampling.

    The complete action space contains more feasible
    allocations than the learning algorithms need to use
    directly.

    The first selected action is the allocation closest to
    equal sharing. Subsequent actions are selected so that
    they are as far as possible from the already selected
    allocations.

    This produces a diverse and reproducible action set.
    """

    if number_of_actions <= 0:
        raise ValueError(
            "NUMBER_OF_ACTIONS must be greater than zero."
        )

    if number_of_actions > len(all_actions):
        raise ValueError(
            f"Requested {number_of_actions} actions, "
            f"but only {len(all_actions)} valid actions exist."
        )

    if number_of_actions == len(all_actions):
        return all_actions.copy()

    action_matrix = np.asarray(
        [
            action_to_array(action)
            for action in all_actions
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Start with the action closest to equal allocation.
    # --------------------------------------------------------

    equal_allocation = np.full(
        len(SLICE_NAMES),
        100.0 / len(SLICE_NAMES),
        dtype=np.float32,
    )

    distances_to_equal = np.linalg.norm(
        action_matrix - equal_allocation,
        axis=1,
    )

    first_index = int(
        np.argmin(distances_to_equal)
    )

    selected_indices = [first_index]

    selected_mask = np.zeros(
        len(all_actions),
        dtype=bool,
    )

    selected_mask[first_index] = True

    minimum_distances = np.linalg.norm(
        action_matrix - action_matrix[first_index],
        axis=1,
    )

    minimum_distances[first_index] = -1.0

    # --------------------------------------------------------
    # Farthest-point sampling
    # --------------------------------------------------------

    while len(selected_indices) < number_of_actions:

        next_index = int(
            np.argmax(minimum_distances)
        )

        if selected_mask[next_index]:
            raise RuntimeError(
                "Duplicate action index selected."
            )

        selected_indices.append(next_index)
        selected_mask[next_index] = True

        distances_to_new_action = np.linalg.norm(
            action_matrix - action_matrix[next_index],
            axis=1,
        )

        minimum_distances = np.minimum(
            minimum_distances,
            distances_to_new_action,
        )

        minimum_distances[selected_mask] = -1.0

    selected_actions = [
        all_actions[index]
        for index in selected_indices
    ]

    # Sort so action indices remain deterministic.
    selected_actions.sort(
        key=lambda action: tuple(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )
    )

    return selected_actions


# ============================================================
# VALIDATE FINAL ACTION SET
# ============================================================

def validate_actions(
    actions: list[dict[str, int]],
    expected_count: int,
    step_percent: int,
    minimum_percent: int,
) -> None:
    """
    Validate the final common action list.
    """

    if len(actions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} actions, "
            f"but generated {len(actions)}."
        )

    action_tuples = [
        tuple(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )
        for action in actions
    ]

    if len(set(action_tuples)) != len(action_tuples):
        raise ValueError(
            "Duplicate resource-allocation actions were generated."
        )

    for action_index, action in enumerate(actions):

        if set(action.keys()) != set(SLICE_NAMES):
            raise ValueError(
                f"Action {action_index} contains invalid slice names."
            )

        total_percent = sum(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )

        if total_percent != 100:
            raise ValueError(
                f"Action {action_index} totals "
                f"{total_percent}%, not 100%."
            )

        for slice_name in SLICE_NAMES:

            allocation = action[slice_name]

            if allocation < minimum_percent:
                raise ValueError(
                    f"Action {action_index}: {slice_name} "
                    f"has only {allocation}%."
                )

            if allocation > 100:
                raise ValueError(
                    f"Action {action_index}: {slice_name} "
                    f"exceeds 100%."
                )

            if allocation % step_percent != 0:
                raise ValueError(
                    f"Action {action_index}: {slice_name} "
                    f"is not a multiple of "
                    f"{step_percent}%."
                )


# ============================================================
# CREATE COMMON ACTION SET
# ============================================================

def create_common_actions() -> list[dict[str, int]]:
    """
    Generate exactly the configured number of common actions.

    The final action set is shared by:
        - D3QN
        - PPO
        - conventional baseline, where applicable
    """

    number_of_actions = int(
        CONFIG["NUMBER_OF_ACTIONS"]
    )

    step_percent = int(
        CONFIG["ACTION_STEP_PERCENT"]
    )

    minimum_percent = int(
        CONFIG["MINIMUM_SHARE_PERCENT"]
    )

    all_valid_actions = generate_all_valid_actions(
        step_percent=step_percent,
        minimum_percent=minimum_percent,
    )

    selected_actions = select_representative_actions(
        all_actions=all_valid_actions,
        number_of_actions=number_of_actions,
    )

    validate_actions(
        actions=selected_actions,
        expected_count=number_of_actions,
        step_percent=step_percent,
        minimum_percent=minimum_percent,
    )

    return selected_actions


# ============================================================
# FINAL COMMON ACTION LIST
# ============================================================

ACTIONS = create_common_actions()


# ============================================================
# ACCESS ONE ACTION
# ============================================================

def get_action(
    action_index: int,
) -> dict[str, int]:
    """
    Return one percentage-allocation action.
    """

    if not isinstance(
        action_index,
        (int, np.integer),
    ):
        raise TypeError(
            "action_index must be an integer."
        )

    action_index = int(action_index)

    if not 0 <= action_index < len(ACTIONS):
        raise IndexError(
            f"action_index must be between 0 and "
            f"{len(ACTIONS) - 1}."
        )

    return ACTIONS[action_index].copy()


# ============================================================
# ACTION AS ARRAY
# ============================================================

def get_action_array(
    action_index: int,
) -> np.ndarray:
    """
    Return one action as:

        [eMBB, URLLC1, URLLC2, BE1]
    """

    return action_to_array(
        get_action(action_index)
    )


# ============================================================
# ACTION AS FRACTIONAL SHARES
# ============================================================

def get_action_shares(
    action_index: int,
) -> dict[str, float]:
    """
    Convert percentage allocations into fractions.

    Example:
        40% -> 0.40
    """

    action = get_action(action_index)

    return {
        slice_name: (
            action[slice_name] / 100.0
        )
        for slice_name in SLICE_NAMES
    }


# ============================================================
# ACTION AS RESOURCE-BLOCK ALLOCATION
# ============================================================

def get_rb_allocation(
    action_index: int,
    total_rbs: int | float | None = None,
) -> dict[str, int]:
    """
    Convert a percentage allocation into integer RB counts.

    Example with TOTAL_RB = 100:

        40% -> 40 RBs
        25% -> 25 RBs
        20% -> 20 RBs
        15% -> 15 RBs
    """

    if total_rbs is None:
        total_rbs = int(
            CONFIG["TOTAL_RB"]
        )

    total_rbs_float = float(total_rbs)

    if total_rbs_float <= 0:
        raise ValueError(
            "total_rbs must be greater than zero."
        )

    # The current experiment uses integer RB counts.
    total_rbs_int = int(
        round(total_rbs_float)
    )

    if not np.isclose(
        total_rbs_float,
        total_rbs_int,
    ):
        raise ValueError(
            "TOTAL_RB must be an integer when using "
            "integer RB allocation."
        )

    shares = get_action_shares(
        action_index
    )

    rb_allocation = {
        slice_name: int(
            round(
                shares[slice_name]
                * total_rbs_int
            )
        )
        for slice_name in SLICE_NAMES
    }

    # --------------------------------------------------------
    # Protect against rounding errors.
    # --------------------------------------------------------

    allocated_total = sum(
        rb_allocation.values()
    )

    difference = (
        total_rbs_int - allocated_total
    )

    if difference != 0:
        # Add/subtract any rounding difference from the
        # slice with the largest percentage allocation.
        largest_slice = max(
            SLICE_NAMES,
            key=lambda slice_name: shares[
                slice_name
            ],
        )

        rb_allocation[
            largest_slice
        ] += difference

    if sum(rb_allocation.values()) != total_rbs_int:
        raise RuntimeError(
            "RB allocation does not equal TOTAL_RB."
        )

    if any(
        value < 0
        for value in rb_allocation.values()
    ):
        raise RuntimeError(
            "Negative RB allocation generated."
        )

    return rb_allocation


# ============================================================
# PRINT ACTION-SPACE SUMMARY
# ============================================================

def print_action_summary() -> None:
    """
    Display information about the common action space.
    """

    all_valid_actions = generate_all_valid_actions(
        step_percent=int(
            CONFIG["ACTION_STEP_PERCENT"]
        ),
        minimum_percent=int(
            CONFIG["MINIMUM_SHARE_PERCENT"]
        ),
    )

    print("=" * 70)
    print("COMMON RESOURCE-ALLOCATION ACTION SPACE")
    print("=" * 70)

    print(
        f"Network slices                  : "
        f"{len(SLICE_NAMES)}"
    )

    print(
        f"Total resource blocks           : "
        f"{CONFIG['TOTAL_RB']}"
    )

    print(
        f"Allocation step                 : "
        f"{CONFIG['ACTION_STEP_PERCENT']}%"
    )

    print(
        f"Minimum allocation per slice    : "
        f"{CONFIG['MINIMUM_SHARE_PERCENT']}%"
    )

    print(
        f"Total feasible allocations      : "
        f"{len(all_valid_actions)}"
    )

    print(
        f"Selected common actions         : "
        f"{len(ACTIONS)}"
    )

    print("\nFirst 10 selected actions:")

    for index, action in enumerate(
        ACTIONS[:10]
    ):
        print(
            f"Action {index}: {action}"
        )

    print("\nLast 5 selected actions:")

    start_index = max(
        0,
        len(ACTIONS) - 5,
    )

    for offset, action in enumerate(
        ACTIONS[-5:]
    ):

        action_index = (
            start_index + offset
        )

        print(
            f"Action {action_index}: "
            f"{action}"
        )

    print("\nEqual-allocation check:")

    equal_action = {
        slice_name: 25
        for slice_name in SLICE_NAMES
    }

    if equal_action in ACTIONS:
        equal_index = ACTIONS.index(
            equal_action
        )

        print(
            f"Equal allocation is included "
            f"at action index {equal_index}."
        )
    else:
        print(
            "WARNING: Equal allocation is not "
            "included in the selected action set."
        )

    print("\nExample conversion:")

    example_index = 0

    print(
        f"Action {example_index}: "
        f"{get_action(example_index)}"
    )

    print(
        f"Shares: "
        f"{get_action_shares(example_index)}"
    )

    print(
        f"RB allocation: "
        f"{get_rb_allocation(example_index)}"
    )

    print("=" * 70)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    print_action_summary()