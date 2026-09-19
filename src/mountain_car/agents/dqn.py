"""
Deep Q-Network (DQN) implementation in PyTorch.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : a small fully-connected network that maps state -> Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', terminated) transitions for replay
  - DQNAgent     : the training loop, epsilon-greedy policy, target-net sync
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """EXERCISE 2a: the network that maps a state to one Q-value per action.

    Build a small fully-connected net:

        state_dim -> hidden -> hidden -> action_dim

    with a ReLU after each hidden layer. There is NO activation on the output
    layer: these are Q-values (here they are all negative), not probabilities.

    Tip: nn.Sequential(nn.Linear(...), nn.ReLU(), ...) is the shortest route.
    Tip: remember super().__init__() before assigning any submodule.
    Tip: forward() receives a batch of shape (B, state_dim) and must return
         shape (B, action_dim).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()  # SIEMPRE antes de crear submódulos
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),   # 2 -> 128
            nn.ReLU(),
            nn.Linear(hidden, hidden),      # 128 -> 128
            nn.ReLU(),
            nn.Linear(hidden, action_dim),  # 128 -> 3, sin activación
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
        stickiness: float = 0.9,   # <- nuevo: probabilidad de repetir la última acción exploratoria

    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.hidden = hidden
        self.training_episodes = 0

        env = gym.make(env_id)
        self.state_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)
        
        self.stickiness = stickiness
        self._last_explore_action: int | None = None

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Textbook epsilon-greedy: explore with probability epsilon.

        EXERCISE 3: this is the standard, by-the-book implementation, and it is
        not enough. Once EXERCISE 2 is done, `train dqn` will run happily and
        report a completely flat score, forever, having learned nothing.

        Your job is to work out WHY and fix it. The bug is not in this method's
        code -- it is correct epsilon-greedy. It is in what this exploration
        strategy can actually reach in this particular environment.

        Starting clue: the car is too weak to drive straight up the hill, so it
        has to rock back and forth in sustained runs to build momentum. Every
        call below draws a completely fresh random action. Can a policy built
        from independent per-step coin flips produce a sustained run?

        EXERCISES.md has the full investigation and a ladder of further clues,
        from gentle to nearly-the-answer -- take only as many as you need. Try
        to diagnose it from your own measurements first.
        """
        if not deterministic and random.random() < self.epsilon:
            # Exploración "pegajosa": con cierta probabilidad, repite la última
            #acción exploratoria en vez de sortear una nueva -> produce rachas
            # sostenidas (empuje izquierda/derecha) en vez de temblor aleatorio.
            if self._last_explore_action is not None and random.random() < self.stickiness:
                return self._last_explore_action
            action = random.randrange(self.action_dim)
            self._last_explore_action = action
            return action

        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(dim=1).item())

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── learning step ─────────────────────────────────────────────────

    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.array(x), dtype=dtype, device=self.device)

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and take one gradient step.

        Returns the batch loss value.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)

        states_t = self._tensor(states)
        actions_t = self._tensor(actions, torch.int64).unsqueeze(1)
        rewards_t = self._tensor(rewards).unsqueeze(1)
        next_states_t = self._tensor(next_states)
        terminateds_t = self._tensor(terminateds).unsqueeze(1)

        # 1. Q(s, a) de la red ONLINE, solo para las acciones tomadas -> (B, 1)
        current_q = self.q_net(states_t).gather(1, actions_t)

        # 2. max_a' Q_target(s', a') de la red TARGET, sin gradiente
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

            # 3. Target de Bellman, anulando el bootstrap donde terminated=1
            target_q = rewards_t + self.gamma * next_q * (1.0 - terminateds_t)

        # 4. Paso de gradiente
        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            self._last_explore_action = None
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store `terminated`, not `done`: hitting the 200-step time
                # limit is not a real terminal state, so we must keep
                # bootstrapping through it.
                self.buffer.push(obs, action, float(reward), next_obs, terminated)
                self._learn()

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    _HPARAMS = (
        "env_id",
        "lr",
        "gamma",
        "epsilon_end",
        "epsilon_decay",
        "batch_size",
        "buffer_capacity",
        "target_update_freq",
        "hidden",
        "stickiness",   # <- agregado
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_net_state"] = self.q_net.state_dict()
        data["optimizer_state"] = self.optimizer.state_dict()
        data["epsilon"] = self.epsilon
        data["training_episodes"] = self.training_episodes
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        # The target net starts as a copy of the online net; it re-syncs during
        # training anyway, so there is no need to persist it separately.
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["q_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Device            : {self.device}"
        )
