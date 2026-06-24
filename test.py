from src.environment import SlicingEnv

env = SlicingEnv()
state = env.reset()

print("Initial State:", state)

for t in range(10):
    next_state, reward, done, _ = env.step(0)
    print(f"\nStep {t}")
    print("State:", next_state)
    print("Reward:", reward)
    print("Done:", done)
 