import csv
from pathlib import Path

from mountain_car.agents.dqn import DQNAgent

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def train_and_log(agent: DQNAgent, episodes: int, csv_path: Path, log_interval: int = 50) -> None:
    rewards = agent.train(total_episodes=episodes, log_interval=log_interval)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([episode, reward])
    print(f"Guardado: {csv_path} ({len(rewards)} filas)")


if __name__ == "__main__":
    # Ejercicio 2: exploración epsilon-greedy pura (sin memoria de la acción anterior)
    agent_ex2 = DQNAgent("MountainCar-v0", stickiness=0.0)
    train_and_log(agent_ex2, episodes=1000, csv_path=LOG_DIR / "ejercicio2_dqn.csv")

    # Ejercicio 3: exploración pegajosa (el fix)
    agent_ex3 = DQNAgent("MountainCar-v0", stickiness=0.9)
    train_and_log(agent_ex3, episodes=2500, csv_path=LOG_DIR / "ejercicio3_dqn.csv")