from pathlib import Path

import gymnasium as gym

from mountain_car.agents import QLearningAgent

SAVE_PATH = Path("saves/qlearning_mountaincar.pkl")
VIDEO_DIR = Path("saves/videos/qlearning")
NUM_EPISODES = 3

agent = QLearningAgent.load(SAVE_PATH)

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=str(VIDEO_DIR),
    name_prefix="qlearning",
    episode_trigger=lambda ep: True,  # graba todos los episodios pedidos
)

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        total_reward += reward
    print(f"Episodio {ep + 1}: reward = {total_reward:.1f}")

env.close()
print(f"Videos guardados en {VIDEO_DIR}")