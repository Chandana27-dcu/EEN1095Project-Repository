from __future__ import annotations

from itertools import product

import numpy as np

from src.config_d3q import CONFIG


SLICE_NAMES = tuple(CONFIG["SLICES"])


def generate_all_valid_actions(
    step_percent: int,
    minimum_percent: int,
) -> list[dict[str, int]]:
    """
    Generate all valid percentage-based resource allocations.

    Conditions:
        1. Every allocation is a multiple of step_percent.
        2. Every slice receives at least minimum_percent.
        3. The four slice allocations total exactly 100%.

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

    valid_actions.sort(
        key=lambda action: tuple(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )
    )

    return valid_actions


def action_to_array(
    action: dict[str, int],
) -> np.ndarray:
    """
    Convert an allocation dictionary to a NumPy array.
    """

    return np.asarray(
        [
            action[slice_name]
            for slice_name in SLICE_NAMES
        ],
        dtype=np.float32,
    )


def select_representative_actions(
    all_actions: list[dict[str, int]],
    number_of_actions: int,
) -> list[dict[str, int]]:
    """
    Select a diverse subset using farthest-point sampling.

    The complete 5%-step action space contains more than 155
    combinations. This method selects 155 representative actions
    distributed across the allocation space.
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

    # Begin with the action nearest to equal allocation.
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

    # Sort to ensure action indices remain reproducible.
    selected_actions.sort(
        key=lambda action: tuple(
            action[slice_name]
            for slice_name in SLICE_NAMES
        )
    )

    return selected_actions


def validate_actions(
    actions: list[dict[str, int]],
    expected_count: int,
    step_percent: int,
    minimum_percent: int,
) -> None:
    """
    Validate the final action list.
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
            "Duplicate D3QN actions were generated."
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
                    f"is not a multiple of {step_percent}%."
                )


def create_d3q_actions() -> list[dict[str, int]]:
    """
    Generate exactly the configured number of D3QN actions.
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


# Final action list used by the D3QN environment and agent.
ACTIONS = create_d3q_actions()


def get_action(
    action_index: int,
) -> dict[str, int]:
    """
    Return one percentage-allocation action.
    """

    if not isinstance(action_index, int):
        raise TypeError(
            "action_index must be an integer."
        )

    if not 0 <= action_index < len(ACTIONS):
        raise IndexError(
            f"action_index must be between 0 and "
            f"{len(ACTIONS) - 1}."
        )

    return ACTIONS[action_index].copy()


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


def get_action_shares(
    action_index: int,
) -> dict[str, float]:
    """
    Convert percentage allocations into fractions from 0 to 1.

    Example:
        40% becomes 0.40.
    """

    action = get_action(action_index)

    return {
        slice_name: (
            action[slice_name] / 100.0
        )
        for slice_name in SLICE_NAMES
    }


def get_rb_allocation(
    action_index: int,
    total_rbs: int | float | None = None,
) -> dict[str, float]:
    """
    Convert a percentage action into resource-block allocations.
    """

    if total_rbs is None:
        total_rbs = float(
            CONFIG["TOTAL_RB"]
        )

    total_rbs = float(total_rbs)

    if total_rbs <= 0:
        raise ValueError(
            "total_rbs must be greater than zero."
        )

    shares = get_action_shares(
        action_index
    )

    return {
        slice_name: (
            shares[slice_name] * total_rbs
        )
        for slice_name in SLICE_NAMES
    }


def print_action_summary() -> None:
    """
    Display basic information about the action space.
    """

    all_valid_count = len(
        generate_all_valid_actions(
            step_percent=int(
                CONFIG["ACTION_STEP_PERCENT"]
            ),
            minimum_percent=int(
                CONFIG["MINIMUM_SHARE_PERCENT"]
            ),
        )
    )

    print(
        f"Total valid allocation combinations: "
        f"{all_valid_count}"
    )

    print(
        f"Selected D3QN actions: {len(ACTIONS)}"
    )

    print("\nFirst 10 selected actions:")

    for index, action in enumerate(
        ACTIONS[:10]
    ):
        print(
            f"Action {index}: {action}"
        )

    print("\nLast 5 selected actions:")

    start_index = len(ACTIONS) - 5

    for offset, action in enumerate(
        ACTIONS[-5:]
    ):
        action_index = start_index + offset

        print(
            f"Action {action_index}: {action}"
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


if __name__ == "__main__":
    print_action_summary()