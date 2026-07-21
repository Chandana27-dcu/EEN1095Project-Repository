from __future__ import annotations

import torch
from torch import nn

from src.config_d3q import CONFIG


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network used by the Double Dueling DQN agent.

    Input:
        Normalized state vector of size 24.

    Shared hidden layers:
        24 -> 128 -> 128

    Output:
        One Q-value for each of the 155 allocation actions.

    The network separates into:
        - Value stream: estimates V(s)
        - Advantage stream: estimates A(s, a)
    """

    def __init__(
        self,
        state_size: int | None = None,
        action_size: int | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()

        if state_size is None:
            state_size = int(CONFIG["STATE_SIZE"])

        if action_size is None:
            action_size = int(CONFIG["NUMBER_OF_ACTIONS"])

        if hidden_size is None:
            hidden_size = int(CONFIG["HIDDEN_SIZE"])

        if state_size <= 0:
            raise ValueError(
                "state_size must be greater than zero."
            )

        if action_size <= 0:
            raise ValueError(
                "action_size must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero."
            )

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Shared feature-extraction layers.
        self.feature_layer = nn.Sequential(
            nn.Linear(
                self.state_size,
                self.hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                self.hidden_size,
                self.hidden_size,
            ),
            nn.ReLU(),
        )

        # State-value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                self.hidden_size,
                1,
            ),
        )

        # Action-advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
            ),
            nn.ReLU(),

            nn.Linear(
                self.hidden_size,
                self.action_size,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize linear layers using Kaiming initialization.
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return Q-values for all allocation actions.

        Input shape:
            [batch_size, state_size]

        Output shape:
            [batch_size, action_size]
        """

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected state size {self.state_size}, "
                f"but received {state.shape[-1]}."
            )

        features = self.feature_layer(state)

        state_value = self.value_stream(
            features
        )

        advantages = self.advantage_stream(
            features
        )

        centered_advantages = (
            advantages
            - advantages.mean(
                dim=1,
                keepdim=True,
            )
        )

        q_values = (
            state_value
            + centered_advantages
        )

        return q_values


def create_network(
    device: torch.device | str | None = None,
) -> DuelingQNetwork:
    """
    Create and optionally move the network to a device.
    """

    network = DuelingQNetwork()

    if device is not None:
        network = network.to(device)

    return network


if __name__ == "__main__":
    network = DuelingQNetwork()

    test_state = torch.zeros(
        1,
        CONFIG["STATE_SIZE"],
        dtype=torch.float32,
    )

    with torch.no_grad():
        q_values = network(test_state)

    print(network)

    print(
        "\nInput shape:",
        tuple(test_state.shape),
    )

    print(
        "Output shape:",
        tuple(q_values.shape),
    )

    print(
        "Expected output actions:",
        CONFIG["NUMBER_OF_ACTIONS"],
    )

    print(
        "Predicted action index:",
        int(torch.argmax(q_values, dim=1).item()),
    )