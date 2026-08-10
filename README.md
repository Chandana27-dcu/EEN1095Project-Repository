# DRL-Assisted Dynamic Network Slicing Management in Beyond 5G/6G Networks

## Project Overview

This project investigates **Deep Reinforcement Learning (DRL)-assisted dynamic radio resource allocation** for network slicing in Beyond 5G (B5G) / 6G networks.

The objective is to dynamically allocate a fixed pool of **100 Resource Blocks (RBs)** among four network slices while adapting to changing traffic conditions and Quality of Service (QoS) requirements.

The proposed method is a **Double Dueling Deep Q-Network (D3QN)**. Its performance is compared with:

- **Proximal Policy Optimization (PPO)** — DRL comparison method
- **Static Equal Allocation (SEA)** — conventional non-learning baseline

The final implementation was corrected and validated to ensure that all methods are evaluated under a common and fair simulation setup.

---

## Network Slices

The environment contains four slices:

| Slice | General Role |
|---|---|
| `eMBB` | Enhanced Mobile Broadband / high-data-rate traffic |
| `URLLC1` | Low-latency and reliability-sensitive traffic |
| `URLLC2` | Periodic URLLC-style traffic |
| `BE1` | Best-effort traffic |

Total available resources:

```text
TOTAL_RB = 100
```

---

## Final Common Simulation Setup

| Parameter | Value |
|---|---:|
| Number of slices | 4 |
| Total Resource Blocks | 100 |
| Bits per RB | 300 |
| State dimension | 24 |
| Discrete action count | 155 |
| Episode length | 500 steps |
| Reward throughput weight | 0.40 |
| Reward latency weight | 0.35 |
| Reward PLR weight | 0.25 |

The corrected D3QN and PPO implementations use the same traffic definitions, channel model, action mapping, reward logic and evaluation conditions.

---

## State Space

The state contains **24 normalized values**:

```text
4 slices × 6 features per slice = 24 state features
```

The per-slice state representation contains:

1. Throughput
2. Latency
3. Packet Loss Ratio (PLR)
4. Queue occupancy
5. Channel condition
6. Traffic load

---

## Action Space

The common resource-allocation action space is:

```text
Discrete(155)
```

Each action represents a valid percentage-based allocation of the 100 RBs among the four slices.

### Action generation constraints

- Allocation step: **5%**
- Minimum share per slice: **5%**
- All slice allocations must sum to **100%**

The full constrained search produces:

```text
969 feasible allocations
```

A representative set of **155 actions** is then selected using **deterministic farthest-point sampling**.

The selection procedure:

1. Generate all feasible allocations.
2. Start with the allocation closest to equal sharing.
3. Iteratively select the allocation farthest from the already selected set.
4. Continue until 155 representative actions are obtained.

This is an engineering trade-off between action-space coverage and computational complexity. The value 155 is **not claimed to be mathematically optimal**.

### Static Equal Allocation

Action index:

```text
95
```

maps to:

```text
eMBB   = 25 RB
URLLC1 = 25 RB
URLLC2 = 25 RB
BE1    = 25 RB
```

This action is used by the Static Equal Allocation baseline.

---

## Traffic Scenarios

### Low Traffic

| Slice | Traffic Setting | Packet Size |
|---|---:|---:|
| eMBB | λ = 3 | 16000 bits |
| URLLC1 | λ = 2 | 2400 bits |
| URLLC2 | Period = 2 | 2400 bits |
| BE1 | λ = 1 | 12000 bits |

### Medium Traffic

| Slice | Traffic Setting | Packet Size |
|---|---:|---:|
| eMBB | λ = 6 | 16000 bits |
| URLLC1 | λ = 4 | 2400 bits |
| URLLC2 | Period = 1 | 2400 bits |
| BE1 | λ = 3 | 12000 bits |

### High Traffic

| Slice | Traffic Setting | Packet Size |
|---|---:|---:|
| eMBB | λ = 10 | 16000 bits |
| URLLC1 | λ = 7 | 2400 bits |
| URLLC2 | Period = 1 | 2400 bits |
| BE1 | λ = 6 | 12000 bits |

Traffic generation:

- eMBB — Poisson arrivals
- URLLC1 — Poisson arrivals
- URLLC2 — periodic arrivals
- BE1 — Poisson arrivals

---

# Proposed D3QN Method

The proposed method uses a **Double Dueling Deep Q-Network (D3QN)**.

## Dueling Architecture

```text
24-state vector
      |
      v
Shared feature layers
      |
  +---+---+
  |       |
  v       v
Value   Advantage
V(s)    A(s,a)
  |       |
  +---+---+
      |
      v
   Q(s,a)
      |
      v
155 allocation actions
```

The dueling combination follows the standard form:

```text
Q(s,a) = V(s) + A(s,a) - mean(A)
```

## Double DQN

The Double-DQN logic separates next-action selection and target-action evaluation using the online and target networks. This helps reduce Q-value overestimation.

## Prioritized Experience Replay

Training uses a `PrioritizedReplayBuffer`, allowing more informative experiences to be sampled more frequently.

---

# PPO Comparison Method

**Proximal Policy Optimization (PPO)** is used as the second DRL method.

PPO provides a policy-gradient / actor-critic comparison against the value-based D3QN approach.

The final PPO implementation uses the same corrected state representation, action space, traffic scenarios and environment assumptions.

---

# Static Equal Allocation Baseline

The conventional non-learning baseline is:

```text
Static Equal Allocation (SEA)
```

It always applies:

```text
25 / 25 / 25 / 25
```

RB allocation and does not adapt to the current network state.

---

# Training Methodology

The final methodology is:

```text
Train D3QN on Medium Traffic
              +
Train PPO on Medium Traffic
              |
              v
       Freeze trained models
              |
              v
Evaluate the same trained models on:
        Low / Medium / High
```

Final trained models:

```text
models/d3qn_medium_load.pth
models/ppo_medium_load.zip
```

---

# Evaluation Metrics

The final evaluation compares D3QN, PPO and SEA using:

- Reward
- Throughput
- Latency
- Jitter
- Packet Loss Ratio (PLR)

General interpretation:

```text
Reward      ↑ higher is better
Throughput  ↑ higher is better
Latency     ↓ lower is better
Jitter      ↓ lower is better
PLR         ↓ lower is better
```

Latency should be interpreted together with PLR because delay statistics are calculated for packets that are successfully served.

---

# Final Evaluation

The common final evaluation is implemented in:

```text
src/evaluate_all.py
```

It evaluates all three methods across Low, Medium and High traffic using multiple evaluation seeds.

Final result files:

```text
results/final_all_methods_detailed.csv
results/final_all_methods_run_summary.csv
results/final_all_methods_summary.csv
```

Final plots:

```text
results/final_plots/01_reward_vs_traffic.png
results/final_plots/02_throughput_vs_traffic.png
results/final_plots/03_latency_vs_traffic.png
results/final_plots/04_jitter_vs_traffic.png
results/final_plots/05_plr_vs_traffic.png
```

Training convergence:

```text
results/d3qn_vs_ppo_convergence.png
```

---

# Main Result Interpretation

The final experiments show that there is no single method that dominates every metric and every traffic condition.

Main observations:

- D3QN achieves the strongest aggregate reward under Low traffic.
- D3QN and PPO are close under Medium traffic, with D3QN slightly higher in the final experiment.
- PPO performs strongly on several individual QoS metrics.
- PPO obtains the strongest reward under High traffic.
- SEA provides a useful conventional benchmark but cannot adapt to changing traffic conditions.

The project therefore evaluates **performance trade-offs**, rather than claiming that one DRL method is universally superior.

---

# Project Structure

```text
EEN1095Project-Repository/
|
|-- archive/
|   `-- pre_supervisor_cleanup_20260810/
|       `-- previous implementation and old experimental outputs
|
|-- models/
|   |-- d3qn_medium_load.pth
|   `-- ppo_medium_load.zip
|
|-- results/
|   |-- final_plots/
|   |-- baseline_medium_results.csv
|   |-- d3qn_medium_training_history.csv
|   |-- d3qn_medium_training_losses.npy
|   |-- d3qn_medium_training_rewards.npy
|   |-- d3qn_vs_ppo_convergence.png
|   |-- final_all_methods_detailed.csv
|   |-- final_all_methods_run_summary.csv
|   |-- final_all_methods_summary.csv
|   `-- ppo_medium_training_history.csv
|
|-- src/
|   |-- actions_common.py
|   |-- baseline_fixed.py
|   |-- config_d3q.py
|   |-- config_ppo.py
|   |-- environment_d3q.py
|   |-- environment_ppo.py
|   |-- evaluate_all.py
|   |-- evaluate_baseline.py
|   |-- network_d3q.py
|   |-- plot_convergence.py
|   |-- plot_final_results.py
|   |-- prioritized_replay.py
|   |-- traffic_common.py
|   |-- train_d3q.py
|   `-- train_ppo.py
|
|-- test_all_methods_fairness.py
|-- .gitignore
`-- README.md
```

---

# Source File Description

| File | Purpose |
|---|---|
| `src/actions_common.py` | Generates and validates the common 155-action allocation space |
| `src/baseline_fixed.py` | Static Equal Allocation baseline |
| `src/config_d3q.py` | D3QN configuration |
| `src/config_ppo.py` | PPO configuration |
| `src/environment_d3q.py` | D3QN network slicing environment |
| `src/environment_ppo.py` | PPO-compatible environment |
| `src/evaluate_all.py` | Common evaluation for D3QN, PPO and SEA |
| `src/evaluate_baseline.py` | Standalone baseline evaluation |
| `src/network_d3q.py` | Dueling Q-Network architecture |
| `src/plot_convergence.py` | D3QN/PPO convergence plotting |
| `src/plot_final_results.py` | Final performance plots |
| `src/prioritized_replay.py` | Prioritized Experience Replay |
| `src/traffic_common.py` | Common traffic generation |
| `src/train_d3q.py` | D3QN training |
| `src/train_ppo.py` | PPO training |
| `test_all_methods_fairness.py` | Final fairness and environment sanity test |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Chandana27-dcu/EEN1095Project-Repository.git
cd EEN1095Project-Repository
git switch final-corrected-project
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

If `requirements.txt` is available:

```powershell
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
matplotlib
torch
gymnasium
stable-baselines3
```

---

# Validation and Testing

## Compile all final Python files

```powershell
Get-ChildItem "src\*.py" | ForEach-Object {
    python -m py_compile $_.FullName
}
```

No output indicates successful compilation.

## Run the final fairness test

```powershell
python test_all_methods_fairness.py
```

The validated implementation should report:

```text
D3QN State Size     : 24
PPO State Size      : 24
Baseline State Size : 24

D3QN Action Space : Discrete(155)
PPO Action Space  : Discrete(155)

Baseline Action   : 95
```

For equal allocation:

```text
D3QN RBs     : {'eMBB': 25, 'URLLC1': 25, 'URLLC2': 25, 'BE1': 25}
PPO RBs      : {'eMBB': 25, 'URLLC1': 25, 'URLLC2': 25, 'BE1': 25}
Baseline RBs : {'eMBB': 25, 'URLLC1': 25, 'URLLC2': 25, 'BE1': 25}
```

A seeded validation produced the same reward for all three methods when the same allocation was applied:

```text
0.5452083333333333
```

## Validate the action space

```powershell
python -m src.actions_common
```

## Load the trained D3QN model

```powershell
python -c "import torch; from src.network_d3q import DuelingQNetwork; c=torch.load('models/d3qn_medium_load.pth',map_location='cpu'); n=DuelingQNetwork(state_size=c['state_size'],action_size=c['action_size'],hidden_size=c['hidden_size']); n.load_state_dict(c['model_state_dict']); print('D3QN MODEL OK')"
```

## Load the trained PPO model

```powershell
python -c "from stable_baselines3 import PPO; PPO.load('models/ppo_medium_load.zip'); print('PPO MODEL OK')"
```

---

# Running the Project

## Evaluate the baseline

```powershell
python -m src.evaluate_baseline
```

## Evaluate D3QN, PPO and SEA together

```powershell
python -m src.evaluate_all
```

## Generate the convergence plot

```powershell
python -m src.plot_convergence
```

## Generate the final comparison plots

```powershell
python -m src.plot_final_results
```

Final plot color convention:

```text
D3QN → Blue
PPO  → Orange
SEA  → Green
```

---

# Reproducibility Notes

Use the final trained Medium-traffic models:

```text
models/d3qn_medium_load.pth
models/ppo_medium_load.zip
```

Do not use archived Low/High pre-correction models when reproducing the final comparison.

---

# Research Scope

This work is a simulation-based engineering study of DRL-assisted resource allocation.

The project does not claim that D3QN is universally superior to all other DRL methods. It investigates:

- dynamic allocation behavior,
- DRL-vs-static allocation differences,
- D3QN-vs-PPO trade-offs,
- generalization from Medium training to Low and High traffic,
- QoS behavior under different traffic loads.

---

# Limitations and Future Work

Possible extensions include:

- NS-3 / 5G NR integration
- more detailed 3GPP channel models
- user mobility
- additional network slices
- continuous action-space DRL
- adaptive or multi-objective reward design
- SAC / TD3 / multi-agent reinforcement learning
- real-time or hardware/testbed validation
