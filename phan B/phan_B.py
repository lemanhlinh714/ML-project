import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================================================
# 1. MÔI TRƯỜNG GRID 5x5 (GIỮ NGUYÊN THEO ĐỀ BÀI)
# =========================================================

class WarehouseEnv:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)   # (1,1)
        self.goal = (4, 4)    # (5,5)

        # obstacle (chuyển sang index 0-based)
        self.obstacles = {(1,1), (1,3), (2,2), (3,1)}

        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        dx, dy = actions[action]
        nx, ny = x + dx, y + dy

        # check boundary
        if nx < 0 or ny < 0 or nx >= 5 or ny >= 5:
            nx, ny = x, y

        # obstacle
        if (nx, ny) in self.obstacles:
            reward = -10
            done = True
            return (nx, ny), reward, done

        # goal
        if (nx, ny) == self.goal:
            return (nx, ny), 100, True

        # normal step
        return (nx, ny), -1, False


# =========================================================
# 2. EPSILON STRATEGIES
# =========================================================

def epsilon_fixed(eps):
    return lambda t: eps

def epsilon_linear(eps_start=1.0, eps_end=0.05, decay_steps=1000):
    def fn(t):
        return max(eps_end, eps_start - t / decay_steps * (eps_start - eps_end))
    return fn

def epsilon_exponential(eps0=1.0, decay=0.995):
    return lambda t: eps0 * (decay ** t)


# =========================================================
# 3. Q-LEARNING
# =========================================================

def train_q_learning(env, alpha, gamma, epsilon_fn, episodes=1000):

    Q = defaultdict(lambda: np.zeros(4))
    rewards_per_episode = []

    steps_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            eps = epsilon_fn(ep)

            # epsilon-greedy
            if np.random.rand() < eps:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)

            # update Q
            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            total_reward += reward
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

    return Q, rewards_per_episode, steps_per_episode


# =========================================================
# 4. POLICY MAP
# =========================================================

def extract_policy(Q):
    policy = np.empty((5,5), dtype=str)

    arrows = {0:'↑', 1:'↓', 2:'←', 3:'→'}

    for i in range(5):
        for j in range(5):
            if (i,j) in [(1,1),(1,3),(2,2),(3,1)]:
                policy[i,j] = "X"
            elif (i,j) == (4,4):
                policy[i,j] = "G"
            else:
                policy[i,j] = arrows[np.argmax(Q[(i,j)])]

    return policy


def print_policy(policy):
    for row in policy:
        print(" ".join(row))


# =========================================================
# 5. RUN EXPERIMENTS (≥ 4 CONFIGS)
# =========================================================

env = WarehouseEnv()

configs = [
    {"name":"C1", "alpha":0.1, "gamma":0.9, "eps":"fixed"},
    {"name":"C2", "alpha":0.3, "gamma":0.9, "eps":"linear"},
    {"name":"C3", "alpha":0.5, "gamma":0.99, "eps":"exp"},
    {"name":"C4", "alpha":0.3, "gamma":0.5, "eps":"fixed"},
]

def get_eps(name):
    if name == "fixed":
        return epsilon_fixed(0.2)
    if name == "linear":
        return epsilon_linear()
    if name == "exp":
        return epsilon_exponential()

results = {}

plt.figure()

for cfg in configs:
    Q, rewards, steps = train_q_learning(
        env,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon_fn=get_eps(cfg["eps"]),
        episodes=1000
    )

    results[cfg["name"]] = {
        "Q": Q,
        "rewards": rewards,
        "steps": steps,
        "avg_steps": np.mean(steps[-200:])
    }

    # smooth reward
    smooth = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(smooth, label=cfg["name"])

plt.title("Reward trung bình theo episode")
plt.legend()
plt.show()


# =========================================================
# 6. SO SÁNH STEPS
# =========================================================

print("\nSO SÁNH SỐ BƯỚC TRUNG BÌNH:")
for k,v in results.items():
    print(k, ":", v["avg_steps"])


# =========================================================
# 7. POLICY TỐT NHẤT
# =========================================================

best = min(results.items(), key=lambda x: x[1]["avg_steps"])

print("\nCONFIG TỐT NHẤT:", best[0])

policy = extract_policy(best[1]["Q"])
print("\nPOLICY MAP:")
print_policy(policy)