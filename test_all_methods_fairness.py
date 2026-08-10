from src.environment_d3q import NetworkSlicingD3QEnv
from src.environment_ppo import NetworkSlicingPPOEnv
from src.baseline_fixed import StaticEqualAllocationBaseline

import numpy as np


SEED = 42
TEST_ACTION = 95


print("=" * 70)
print("FINAL ENVIRONMENT FAIRNESS CHECK")
print("=" * 70)


# D3QN
d3qn_env = NetworkSlicingD3QEnv()
d3qn_state, _ = d3qn_env.reset(seed=SEED)

(
    d3qn_next_state,
    d3qn_reward,
    _,
    _,
    d3qn_info,
) = d3qn_env.step(TEST_ACTION)


# PPO
ppo_env = NetworkSlicingPPOEnv()
ppo_state, _ = ppo_env.reset(seed=SEED)

(
    ppo_next_state,
    ppo_reward,
    _,
    _,
    ppo_info,
) = ppo_env.step(TEST_ACTION)


# Baseline
baseline = StaticEqualAllocationBaseline()
baseline_env = NetworkSlicingD3QEnv()

baseline_state, _ = baseline_env.reset(seed=SEED)

baseline_action = baseline.select_action(
    baseline_state
)

(
    baseline_next_state,
    baseline_reward,
    _,
    _,
    baseline_info,
) = baseline_env.step(baseline_action)


print()
print("D3QN State Size     :", len(d3qn_state))
print("PPO State Size      :", len(ppo_state))
print("Baseline State Size :", len(baseline_state))

print()
print("D3QN Action Space :", d3qn_env.action_space)
print("PPO Action Space  :", ppo_env.action_space)
print("Baseline Action   :", baseline_action)

print()
print("D3QN RBs     :", d3qn_info["rb_allocation"])
print("PPO RBs      :", ppo_info["rb_allocation"])
print("Baseline RBs :", baseline_info["rb_allocation"])

print()
print("D3QN Reward     :", d3qn_reward)
print("PPO Reward      :", ppo_reward)
print("Baseline Reward :", baseline_reward)

print()
print("-" * 70)

print(
    "D3QN/PPO Initial State Match:",
    np.allclose(
        d3qn_state,
        ppo_state,
    ),
)

print(
    "D3QN/Baseline Initial State Match:",
    np.allclose(
        d3qn_state,
        baseline_state,
    ),
)

print(
    "D3QN/PPO Next State Match:",
    np.allclose(
        d3qn_next_state,
        ppo_next_state,
    ),
)

print(
    "D3QN/Baseline Next State Match:",
    np.allclose(
        d3qn_next_state,
        baseline_next_state,
    ),
)

print(
    "D3QN/PPO Reward Match:",
    np.isclose(
        d3qn_reward,
        ppo_reward,
    ),
)

print(
    "D3QN/Baseline Reward Match:",
    np.isclose(
        d3qn_reward,
        baseline_reward,
    ),
)

print("=" * 70)