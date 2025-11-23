from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import imageio.v2 as imageio


GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 5e-4
ENTROPY_BETA = 1e-3
MAX_EPISODES = 4000
MAX_STEPS = 500
PRINT_INTERVAL = 50
SEED = 2023

torch.manual_seed(SEED)
np.random.seed(SEED)

env = gym.make("CartPole-v1")
env.reset(seed=SEED)
env.action_space.seed(SEED)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(n_observations, n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

returns_history: list[float] = []
recent_returns: deque[float] = deque(maxlen=PRINT_INTERVAL)

for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset(seed=SEED + episode)
    episode_return = 0.0

    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated_flags: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    last_terminated = False
    last_truncated = False

    for _ in range(MAX_STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action).squeeze())
        entropies.append(dist.entropy().squeeze())
        values.append(value.squeeze())

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        terminated_flags.append(torch.tensor(float(terminated), dtype=torch.float32, device=device))

        episode_return += reward
        state = next_state
        last_terminated = terminated
        last_truncated = truncated

        if terminated or truncated:
            break

    with torch.no_grad():
        if last_terminated and not last_truncated:
            next_value = torch.tensor(0.0, device=device)
        else:
            next_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, bootstrap_value = model(next_state_tensor)
            next_value = bootstrap_value.squeeze()

    gae = torch.tensor(0.0, device=device)
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    next_val = next_value

    for reward, terminated_flag, value in zip(
        reversed(rewards), reversed(terminated_flags), reversed(values)
    ):
        delta = reward + GAMMA * next_val * (1.0 - terminated_flag) - value
        gae = delta + GAMMA * GAE_LAMBDA * (1.0 - terminated_flag) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + value)
        next_val = value

    advantages_tensor = torch.stack(advantages)
    returns_tensor = torch.stack(returns)
    values_tensor = torch.stack(values)
    log_probs_tensor = torch.stack(log_probs)
    entropies_tensor = torch.stack(entropies)

    if advantages_tensor.numel() > 1:
        advantages_tensor = (
            advantages_tensor - advantages_tensor.mean()
        ) / (advantages_tensor.std(unbiased=False) + 1e-8)

    policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
    value_loss = 0.5 * (returns_tensor - values_tensor).pow(2).mean()
    entropy_loss = -entropies_tensor.mean()
    loss = policy_loss + value_loss + ENTROPY_BETA * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    returns_history.append(episode_return)
    recent_returns.append(episode_return)

    if episode % PRINT_INTERVAL == 0:
        avg_recent = np.mean(recent_returns)
        print(f"Ep {episode:04d} | Return: {episode_return:.1f} | Avg({PRINT_INTERVAL}): {avg_recent:.1f}")

env.close()


def moving_average(sequence: list[float], window: int = 10) -> np.ndarray:
    if len(sequence) < window:
        return np.array(sequence)
    cumsum = np.cumsum(sequence, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def evaluate_policy(env: gym.Env, policy: nn.Module, episodes: int = 20) -> list[float]:
    policy.eval()
    episode_returns: list[float] = []
    with torch.no_grad():
        for idx in range(episodes):
            state, _ = env.reset(seed=SEED + 10_000 + idx)
            ep_return = 0.0
            for _ in range(MAX_STEPS):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy(state_tensor)
                action = torch.argmax(logits, dim=1).item()
                state, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                if terminated or truncated:
                    break
            episode_returns.append(ep_return)
    policy.train()
    return episode_returns



eval_env = gym.make("CartPole-v1")
eval_env.reset(seed=SEED + 50_000)
eval_env.action_space.seed(SEED + 50_000)
test_returns = evaluate_policy(eval_env, model, episodes=20)
eval_env.close()


def record_policy(policy: nn.Module, episodes: int = 1) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    render_env = gym.make("CartPole-v1", render_mode="rgb_array")
    policy.eval()
    with torch.no_grad():
        for idx in range(episodes):
            state, _ = render_env.reset(seed=SEED + 20_000 + idx)
            for _ in range(MAX_STEPS):
                frame = render_env.render()
                frames.append(np.asarray(frame))
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy(state_tensor)
                action = torch.argmax(logits, dim=1).item()
                state, _, terminated, truncated, _ = render_env.step(action)
                if terminated or truncated:
                    break
    policy.train()
    render_env.close()
    return frames


animation_frames = record_policy(model, episodes=1)


Path("fig").mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(moving_average(returns_history))
plt.xlabel("#Episode")
plt.ylabel("Training Return")
plt.title("A2C Training Performance")
plt.tight_layout()
plt.savefig("fig/a2c_results.pdf")

plt.figure(figsize=(6, 4))
plt.bar(np.arange(len(test_returns)), test_returns)
plt.xlabel("Episode")
plt.ylabel("Test Return")
plt.title("A2C Evaluation Episodes")
plt.tight_layout()
plt.savefig("fig/a2c_test_results.pdf")

if animation_frames:
    imageio.mimsave("fig/a2c_animation.gif", animation_frames, fps=30)

plt.show()