#!/usr/bin/env python3
"""
plot_learning_curves.py

Grafica y compara las curvas de aprendizaje de los 3 ejercicios del repo
(mountain_car): Q-Learning tabular (Ejercicio 1), DQN roto (Ejercicio 2) y
DQN corregido con exploración correlacionada (Ejercicio 3).

Lee los CSV de logging (columnas: episode, reward) que genera el CLI del
proyecto (uv run mountaincar train <agent> ...) y dibuja, para cada uno,
el reward promedio (media móvil) por episodio, para poder comparar los
tres métodos/configuraciones en una sola figura.

Uso:
    python plot_learning_curves.py
    python plot_learning_curves.py --window 50 --out plots/learning_curves.png

Por defecto asume que los CSV están en saves/ (tal como se guardan en el
repo), con estos nombres:
    saves/qlearning_training_log.csv   -> Ejercicio 1 (Q-Learning tabular)
    saves/ejercicio2_dqn.csv           -> Ejercicio 2 (DQN sin corregir)
    saves/ejercicio3_dqn.csv           -> Ejercicio 3 (DQN corregido)

La imagen se guarda en plots/learning_curves.png por defecto, para poder
referenciarla en el README (p.ej. ![Learning curves](plots/learning_curves.png))
y que se vea también en GitHub una vez se suba el repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# episode,reward -> uno de estos por método/config a comparar
CURVES = [
    {
        "path": "saves/qlearning_training_log.csv",
        "label": "Ejercicio 1 · Q-Learning (tabular)",
        "color": "#1f77b4",
    },
    {
        "path": "saves/ejercicio2_dqn.csv",
        "label": "Ejercicio 2 · DQN (exploración uniforme)",
        "color": "#d62728",
    },
    {
        "path": "saves/ejercicio3_dqn.csv",
        "label": "Ejercicio 3 · DQN (exploración corregida)",
        "color": "#2ca02c",
    },
]


def load_curve(path: Path, window: int) -> pd.DataFrame | None:
    """Carga un CSV episode,reward y calcula la media móvil del reward."""
    if not path.exists():
        print(f"[aviso] no se encontró {path}, se omite esa curva")
        return None

    df = pd.read_csv(path)
    df = df.sort_values("episode")
    df["avg_reward"] = df["reward"].rolling(window=window, min_periods=1).mean()
    return df


def plot_curves(window: int, out_path: Path, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    any_curve = False
    for curve in CURVES:
        df = load_curve(root / curve["path"], window)
        if df is None:
            continue
        any_curve = True
        ax.plot(
            df["episode"],
            df["avg_reward"],
            label=curve["label"],
            color=curve["color"],
            linewidth=1.6,
        )

    if not any_curve:
        raise SystemExit(
            "No se encontró ningún CSV. Revisa las rutas en CURVES o usa "
            "--root para apuntar a la carpeta correcta."
        )

    # Referencias del README: -200 = nunca llega a la bandera, -110 = "resuelto"
    ax.axhline(-110, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        ax.get_xlim()[1], -110, "  umbral 'resuelto' (-110)",
        va="bottom", ha="right", fontsize=8, color="gray",
    )

    ax.set_xlabel("Episodio")
    ax.set_ylabel(f"Reward promedio (media móvil, ventana={window})")
    ax.set_title("Curvas de aprendizaje — MountainCar-v0")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Gráfica guardada en {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=int, default=50,
        help="tamaño de la ventana para la media móvil (default: 50)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("plots/learning_curves.png"),
        help="ruta de salida de la imagen (default: plots/learning_curves.png)",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="carpeta raíz del repo desde la que resolver las rutas de saves/ (default: .)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_curves(window=args.window, out_path=args.out, root=args.root)


if __name__ == "__main__":
    main()
