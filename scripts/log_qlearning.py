import csv
import sys
from pathlib import Path

from mountain_car.agents import QLearningAgent

TOTAL_EPISODES = 21000
LOG_INTERVAL = 500
OUT_PATH = Path("saves/qlearning_training_log.csv")

agent = QLearningAgent("MountainCar-v0")
rewards = agent.train(total_episodes=TOTAL_EPISODES, log_interval=LOG_INTERVAL)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward"])
    writer.writerows(enumerate(rewards, start=1))

agent.save(Path("saves/qlearning_mountaincar.pkl"))
print(f"Log guardado en {OUT_PATH} ({len(rewards)} episodios)")