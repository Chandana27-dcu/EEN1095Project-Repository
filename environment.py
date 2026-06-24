import numpy as np
from src.traffic import generate_traffic
from src.reward import compute_reward
from src.metrics import update_metrics
from src.config import CONFIG


class SlicingEnv:
    def __init__(self):
        self.total_RB = CONFIG["TOTAL_RB"]
        self.max_time = CONFIG["MAX_TIME"]
        self.time = 0

        self.slices = ["eMBB", "URLLC1", "URLLC2", "BE1"]
        self.queue = {s: [] for s in self.slices}

        self.metrics = {
            "throughput": {s: 0 for s in self.slices},
            "latency": {s: [] for s in self.slices},
            "plr": {s: 0 for s in self.slices},
            "arrivals": {s: 0 for s in self.slices},
        }

        self.actions = CONFIG["ACTIONS"]

    def reset(self):
        self.time = 0
        for s in self.slices:
            self.queue[s] = []
            self.metrics["throughput"][s] = 0
            self.metrics["latency"][s] = []
            self.metrics["plr"][s] = 0
            self.metrics["arrivals"][s] = 0
        return self.get_state()

    def step(self, action_index):
        self.time += 1

        packets = generate_traffic(self.time)
        for s in self.slices:
            for pkt in packets[s]:
                self.queue[s].append(pkt)
                self.metrics["arrivals"][s] += 1

        allocation = self.actions[action_index]
        bits_per_rb = CONFIG["BITS_PER_RB"]

        for s in self.slices:
            capacity = allocation[s] * bits_per_rb
            new_queue = []

            for pkt in self.queue[s]:
                if pkt["size"] <= capacity:
                    delay = self.time - pkt["arrival"]
                    self.metrics["latency"][s].append(delay)
                    self.metrics["throughput"][s] += pkt["size"]
                    capacity -= pkt["size"]
                else:
                    new_queue.append(pkt)

            self.queue[s] = new_queue

        update_metrics(self.metrics, self.queue, self.time)

        reward = compute_reward(self.metrics)
        next_state = self.get_state()
        done = self.time >= self.max_time

        return next_state, reward, done, {}

    def get_state(self):
        state = []

        for s in self.slices:
            q_len = len(self.queue[s])

            avg_lat = (
                np.mean(self.metrics["latency"][s])
                if len(self.metrics["latency"][s]) > 0
                else 0
            )

            jitter = (
                np.std(self.metrics["latency"][s])
                if len(self.metrics["latency"][s]) > 1
                else 0
            )

            plr = self.metrics["plr"][s]

            state.extend([q_len, avg_lat, jitter, plr])

        return np.array(state, dtype=float)