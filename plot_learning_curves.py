"""
plot_learning_curves.py

Compara las curvas de aprendizaje de los tres agentes entrenados sobre
MountainCar-v0: Q-Learning tabular (Ejercicio 1), DQN con exploracion uniforme
(Ejercicio 2) y DQN con exploracion pegajosa (Ejercicio 3).

Lee los CSV de logging (columnas: episode, reward) que generan los scripts de
scripts/ y produce dos figuras:

  plots/learning_curves.png          paneles apilados (small multiples), un
                                     agente por panel, con el eje Y compartido
                                     y el resultado de evaluacion en el titulo
  plots/learning_curves_overlay.png  las tres curvas superpuestas en un solo
                                     eje, para ver de un vistazo cuantos
                                     episodios necesita cada agente

Los paneles apilados son la figura principal porque los tres agentes se
entrenaron durante ordenes de magnitud distintos (21 000 vs 1 000 vs 2 500
episodios): superpuestos, las dos curvas de DQN quedan comprimidas contra el
margen izquierdo y se pierde el detalle.

Uso:
    uv run python plot_learning_curves.py
    uv run python plot_learning_curves.py --window 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paleta Okabe-Ito: distinguible con deuteranopia y protanopia, a diferencia
# del rojo/verde por defecto de matplotlib.
CURVES = [
    {
        "tag": "ejercicio1_qlearning",
        "label": "Ejercicio 1 · Q-Learning (tabular)",
        "short": "Ej. 1 · Q-Learning",
        "color": "#0072B2",
        # (episodio, desplazamiento en puntos) para la etiqueta directa del overlay
        "anchor": (21_000, (8, 0)),
    },
    {
        "tag": "ejercicio2_dqn",
        "label": "Ejercicio 2 · DQN (exploración uniforme)",
        "short": "Ej. 2 · DQN uniforme",
        "color": "#D55E00",
        "anchor": (800, (10, -6)),
    },
    {
        "tag": "ejercicio3_dqn",
        "label": "Ejercicio 3 · DQN (exploración pegajosa)",
        "short": "Ej. 3 · DQN pegajosa",
        "color": "#009E73",
        "anchor": (2_300, (10, 10)),
    },
]

SOLVED_THRESHOLD = -110.0
LOG_DIR = Path("logs")


def load_curve(root: Path, tag: str, window: int) -> tuple[pd.DataFrame, dict | None] | None:
    """Carga un CSV episode,reward mas su JSON de evaluacion, si existe."""
    path = root / LOG_DIR / f"{tag}.csv"
    if not path.exists():
        print(f"[aviso] no se encontró {path}, se omite esa curva")
        return None

    df = pd.read_csv(path).sort_values("episode")
    df["avg_reward"] = df["reward"].rolling(window=window, min_periods=1).mean()

    meta_path = root / LOG_DIR / f"{tag}_eval.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return df, meta


def _threshold(ax, with_label: bool = True) -> None:
    ax.axhline(SOLVED_THRESHOLD, color="#999999", linestyle="--", linewidth=1, zorder=1)
    if with_label:
        ax.annotate(
            f"umbral “resuelto” ({SOLVED_THRESHOLD:.0f})",
            xy=(0.985, SOLVED_THRESHOLD), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points",
            ha="right", fontsize=8, color="#666666",
        )


def _despine(ax) -> None:
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_panels(drawn: list, window: int, out_path: Path) -> None:
    """Small multiples: un panel por agente, eje Y compartido."""
    fig, axes = plt.subplots(
        len(drawn), 1, figsize=(10, 3.1 * len(drawn)), sharey=True,
    )
    axes = [axes] if len(drawn) == 1 else list(axes)

    for ax, (curve, df, meta) in zip(axes, drawn):
        ax.plot(df["episode"], df["reward"], color=curve["color"],
                linewidth=0.5, alpha=0.15)
        ax.plot(df["episode"], df["avg_reward"], color=curve["color"],
                linewidth=2.0, solid_capstyle="round")

        title = f"{curve['label']}  ·  {len(df):,} episodios".replace(",", " ")
        if meta:
            title += (
                f"  ·  evaluación: {meta['mean_reward']:.1f} ± {meta['std_reward']:.1f}"
                f"  ·  bandera {meta['reached_flag']}/{meta['episodes']}"
            )
        ax.set_title(title, fontsize=10, loc="left", pad=8, color="#333333")

        _threshold(ax, with_label=(ax is axes[0]))
        _despine(ax)
        ax.set_ylim(-207, -75)
        ax.set_ylabel("Reward")

    axes[-1].set_xlabel("Episodio de entrenamiento")
    fig.suptitle(
        f"Curvas de aprendizaje — MountainCar-v0  (media móvil, ventana={window})",
        fontsize=13, x=0.01, ha="left", y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en {out_path}")


def plot_overlay(drawn: list, window: int, out_path: Path) -> None:
    """Las tres curvas superpuestas en un solo eje."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for curve, df, _ in drawn:
        ax.plot(df["episode"], df["avg_reward"], label=curve["label"],
                color=curve["color"], linewidth=2.0, solid_capstyle="round")

    # Etiqueta directa sobre cada curva: la identidad de cada serie no depende
    # solo del color (requisito de accesibilidad).
    for curve, df, _ in drawn:
        episode, offset = curve["anchor"]
        row = df.iloc[(df["episode"] - episode).abs().idxmin()]
        ax.annotate(
            curve["short"],
            xy=(row["episode"], row["avg_reward"]),
            xytext=offset, textcoords="offset points",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    _threshold(ax)
    _despine(ax)
    ax.set_ylim(-207, -90)
    ax.margins(x=0.13)
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel(f"Reward promedio (media móvil, ventana={window})")
    ax.set_title("Curvas de aprendizaje superpuestas — MountainCar-v0",
                 fontsize=13, loc="left", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=50,
                        help="tamaño de la ventana para la media móvil (default: 50)")
    parser.add_argument("--out", type=Path, default=Path("plots/learning_curves.png"),
                        help="ruta de la figura de paneles (default: plots/learning_curves.png)")
    parser.add_argument("--out-overlay", type=Path,
                        default=Path("plots/learning_curves_overlay.png"),
                        help="ruta de la figura superpuesta")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="carpeta raíz del repo (default: .)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    drawn = []
    for curve in CURVES:
        loaded = load_curve(args.root, curve["tag"], args.window)
        if loaded is not None:
            drawn.append((curve, *loaded))

    if not drawn:
        raise SystemExit(
            "No se encontró ningún CSV en logs/. Corre primero los scripts de "
            "entrenamiento (ver README) o usa --root."
        )

    plot_panels(drawn, args.window, args.out)
    plot_overlay(drawn, args.window, args.out_overlay)


if __name__ == "__main__":
    main()
