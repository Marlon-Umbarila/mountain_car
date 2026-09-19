"""Entrena el agente tabular de Q-Learning y deja la evidencia del Ejercicio 1.

Genera:
  logs/ejercicio1_qlearning.csv   reward de cada episodio (episode, reward)
  saves/qlearning_mountaincar.pkl agente entrenado
  logs/ejercicio1_qlearning_eval.json  metricas de evaluacion determinista

Uso:
    uv run python scripts/run_qlearning.py [--episodes 21000] [--eval-episodes 100]
"""
import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents import QLearningAgent

LOG_DIR = Path("logs")
SAVE_PATH = Path("saves/qlearning_mountaincar.pkl")


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
    parser.add_argument("--episodes", type=int, default=21_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()

    agent = QLearningAgent("MountainCar-v0")
    rewards = agent.train(total_episodes=args.episodes, log_interval=args.log_interval)

    LOG_DIR.mkdir(exist_ok=True)
    csv_path = LOG_DIR / "ejercicio1_qlearning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(rewards, start=1))
    print(f"Log guardado en {csv_path} ({len(rewards)} episodios)")

    agent.save(SAVE_PATH)

    metrics = evaluate(agent, args.eval_episodes)
    metrics["training_episodes"] = args.episodes
    metrics["states_visited"] = len(agent.q_table)
    eval_path = LOG_DIR / "ejercicio1_qlearning_eval.json"
    eval_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
