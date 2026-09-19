"""Genera las figuras de evidencia por ejercicio y la tabla comparativa.

Para cada agente dibuja el reward episodio a episodio (crudo + media movil),
marca el mejor episodio y anota el resultado de la evaluacion determinista
guardado por los scripts run_qlearning.py / run_dqn.py.

Salidas:
    plots/ejercicio1_qlearning.png
    plots/ejercicio2_dqn.png
    plots/ejercicio3_dqn.png
    logs/resumen_metricas.csv

Uso:
    uv run python scripts/make_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = Path("logs")
PLOT_DIR = Path("plots")
SOLVED_THRESHOLD = -110.0

# Paleta Okabe-Ito (segura para deuteranopia / protanopia).
RUNS = [
    {
        "tag": "ejercicio1_qlearning",
        "title": "Ejercicio 1 · Q-Learning tabular",
        "color": "#0072B2",
    },
    {
        "tag": "ejercicio2_dqn",
        "title": "Ejercicio 2 · DQN con exploración uniforme",
        "color": "#D55E00",
    },
    {
        "tag": "ejercicio3_dqn",
        "title": "Ejercicio 3 · DQN con exploración pegajosa",
        "color": "#009E73",
    },
]


def make_figure(run: dict, df: pd.DataFrame, meta: dict | None, window: int = 50) -> None:
    df = df.sort_values("episode")
    smooth = df["reward"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        df["episode"], df["reward"],
        color=run["color"], linewidth=0.5, alpha=0.18,
        label="Reward por episodio",
    )
    ax.plot(
        df["episode"], smooth,
        color=run["color"], linewidth=2.0, solid_capstyle="round",
        label=f"Media móvil ({window} episodios)",
    )

    # Mejor episodio de entrenamiento.
    best_idx = df["reward"].idxmax()
    best_ep, best_rw = int(df.loc[best_idx, "episode"]), float(df.loc[best_idx, "reward"])
    ax.scatter([best_ep], [best_rw], s=42, color=run["color"], zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"mejor episodio: {best_rw:.0f}  (ep. {best_ep:,})".replace(",", " "),
        xy=(best_ep, best_rw), xytext=(8, 8), textcoords="offset points",
        fontsize=9, color="#333333",
    )

    ax.axhline(SOLVED_THRESHOLD, color="#999999", linestyle="--", linewidth=1)
    ax.annotate(
        f"umbral “resuelto” ({SOLVED_THRESHOLD:.0f})",
        xy=(0.015, SOLVED_THRESHOLD), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", fontsize=8, color="#666666",
    )

    subtitle = f"{len(df):,} episodios de entrenamiento".replace(",", " ")
    if meta:
        subtitle += (
            f"  ·  evaluación determinista ({meta['episodes']} episodios): "
            f"{meta['mean_reward']:.1f} ± {meta['std_reward']:.1f}  ·  "
            f"llega a la bandera {meta['reached_flag']}/{meta['episodes']}"
        )

    ax.set_title(run["title"], fontsize=13, loc="left", pad=18)
    ax.annotate(
        subtitle, xy=(0, 1.02), xycoords="axes fraction",
        fontsize=9, color="#555555", va="bottom",
    )
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Reward del episodio")
    ax.set_ylim(-207, -75)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{run['tag']}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en {out}")


def main() -> None:
    rows = []
    for run in RUNS:
        csv_path = LOG_DIR / f"{run['tag']}.csv"
        if not csv_path.exists():
            print(f"[aviso] falta {csv_path}, se omite")
            continue
        df = pd.read_csv(csv_path)

        meta_path = LOG_DIR / f"{run['tag']}_eval.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

        make_figure(run, df, meta)

        smooth = df["reward"].rolling(window=50, min_periods=1).mean()
        rows.append({
            "ejercicio": run["title"],
            "episodios_entrenados": len(df),
            "reward_medio_ultimos_50": round(df["reward"].tail(50).mean(), 2),
            "mejor_media_movil_50": round(smooth.max(), 2),
            "mejor_episodio": round(df["reward"].max(), 2),
            "peor_episodio": round(df["reward"].min(), 2),
            "eval_reward_medio": round(meta["mean_reward"], 2) if meta else None,
            "eval_desviacion": round(meta["std_reward"], 2) if meta else None,
            "eval_llega_bandera": (
                f"{meta['reached_flag']}/{meta['episodes']}" if meta else None
            ),
        })

    if rows:
        summary = pd.DataFrame(rows)
        out = LOG_DIR / "resumen_metricas.csv"
        summary.to_csv(out, index=False)
        print(f"\nResumen guardado en {out}\n")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
