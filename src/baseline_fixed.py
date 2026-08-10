from __future__ import annotations


class StaticEqualAllocationBaseline:
    """
    Conventional non-learning baseline.

    The available RBs are divided equally:
        eMBB   = 25%
        URLLC1 = 25%
        URLLC2 = 25%
        BE1    = 25%

    Equal allocation corresponds to action index 95.
    """

    def __init__(self) -> None:
        self.name = "Static Equal Allocation"
        self.action_index = 95

    def select_action(self, state=None) -> int:
        del state
        return self.action_index


if __name__ == "__main__":

    baseline = StaticEqualAllocationBaseline()

    print("=" * 60)
    print("STATIC EQUAL ALLOCATION BASELINE")
    print("=" * 60)

    print("Baseline:", baseline.name)
    print("Selected action:", baseline.action_index)

    print(
        "Allocation:",
        {
            "eMBB": 25,
            "URLLC1": 25,
            "URLLC2": 25,
            "BE1": 25,
        },
    )