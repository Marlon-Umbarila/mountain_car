"""Entrena un agente DQN y deja la evidencia de los Ejercicios 2 y 3.

El unico parametro que separa el Ejercicio 2 del 3 es `stickiness`:
  --stickiness 0.0  exploracion epsilon-greedy uniforme  -> Ejercicio 2 (no aprende)
  --stickiness 0.9  exploracion pegajosa (el fix)        -> Ejercicio 3

Genera:
  logs/<tag>.csv        reward de cada episodio (episode, reward)
  logs/<tag>_eval.json  metricas de evaluacion determinista
  saves/<tag>.pt        agente entrenado

Uso:
    uv run python scripts/run_dqn.py --tag ejercicio3_dqn --stickiness 0.9 --episodes 2500
"""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents.dqn import DQNAgent

LOG_DIR = Path("logs")
SAVE_DIR = Path("saves")


def evaluate(agent, n_episodes: int, seed: int = 0) -> dict:
    """Corre n episodios en modo determinista y resume el desempeno."""
    env = gym.make("MountainCar-v0")
    rewards, reached = [], 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total += reward
        reached += int(terminated)
        rewards.append(total)
    env.close()
    return {
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards)),
        "worst_reward": float(np.min(rewards)),
        "reached_flag": reached,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="nombre base de los archivos de salida")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--stickiness", type=float, default=0.9)
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DQNAgent(
        "MountainCar-v0",
        stickiness=args.stickiness,
        target_update_freq=args.target_update_freq,
        epsilon_decay=args.epsilon_decay,
    )
    rewards = agent.train(total_episodes=args.episodes, log_interval=args.log_interval)

    LOG_DIR.mkdir(exist_ok=True)
    csv_path = LOG_DIR / f"{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(rewards, start=1))
    print(f"Log guardado en {csv_path} ({len(rewards)} episodios)")

    agent.save(SAVE_DIR / f"{args.tag}.pt")

    metrics = evaluate(agent, args.eval_episodes)
    metrics.update(
        training_episodes=args.episodes,
        stickiness=args.stickiness,
        target_update_freq=args.target_update_freq,
        epsilon_decay=args.epsilon_decay,
    )
    (LOG_DIR / f"{args.tag}_eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
