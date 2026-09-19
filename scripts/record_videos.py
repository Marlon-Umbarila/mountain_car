from pathlib import Path

import gymnasium as gym

from mountain_car.agents.dqn import DQNAgent

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)
SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)


def record_last_episode(agent: DQNAgent, name: str, episodes: int = 3) -> None:
    """Corre `episodes` episodios deterministas y graba SOLO el último en video."""
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        name_prefix=name,
        episode_trigger=lambda ep: ep == episodes - 1,  # solo la última corrida
    )
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    print(f"Video guardado en {VIDEO_DIR}/ con prefijo '{name}'")


if __name__ == "__main__":
    # Ejercicio 2: exploración epsilon-greedy pura (sin memoria) -> se queda en -200
    agent_ex2 = DQNAgent("MountainCar-v0", stickiness=0.0)
    agent_ex2.train(total_episodes=1000)
    agent_ex2.save(SAVE_DIR / "dqn_ejercicio2.pt")
    record_last_episode(agent_ex2, name="ejercicio2_dqn", episodes=3)

    # Ejercicio 3: exploración pegajosa (el fix) -> llega a la bandera
    agent_ex3 = DQNAgent("MountainCar-v0", stickiness=0.9)
    agent_ex3.train(total_episodes=2500)
    agent_ex3.save(SAVE_DIR / "dqn_ejercicio3.pt")
    record_last_episode(agent_ex3, name="ejercicio3_dqn", episodes=3)