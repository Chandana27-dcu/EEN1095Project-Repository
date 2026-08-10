from environment_d3q import SlicingEnv

env = SlicingEnv()
state = env.reset()

print("===== INITIAL STATE =====")
print(state)

for t in range(10):
    next_state, reward, done, _ = env.step(0)

    print(f"\n========== STEP {t} ==========")

    print(
        f"eMBB   -> Queue={next_state[0]:.2f}, "
        f"Latency={next_state[1]:.2f}, "
        f"Jitter={next_state[2]:.2f}, "
        f"PLR={next_state[3]:.4f}"
    )

    print(
        f"URLLC1 -> Queue={next_state[4]:.2f}, "
        f"Latency={next_state[5]:.2f}, "
        f"Jitter={next_state[6]:.2f}, "
        f"PLR={next_state[7]:.4f}"
    )

    print(
        f"URLLC2 -> Queue={next_state[8]:.2f}, "
        f"Latency={next_state[9]:.2f}, "
        f"Jitter={next_state[10]:.2f}, "
        f"PLR={next_state[11]:.4f}"
    )

    print(
        f"BE1    -> Queue={next_state[12]:.2f}, "
        f"Latency={next_state[13]:.2f}, "
        f"Jitter={next_state[14]:.2f}, "
        f"PLR={next_state[15]:.4f}"
    )

    print(f"\nReward = {reward:.4f}")
    print(f"Done   = {done}")

    if done:
        break