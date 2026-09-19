# Esquemas del ciclo de entrenamiento

Esta carpeta guarda los dos esquemas dibujados a mano que documentan el ciclo
de entrenamiento de cada agente. El README principal los referencia así:

```
docs/esquema_qlearning.png
docs/esquema_dqn.png
```

Guarda aquí las imágenes con exactamente esos nombres (PNG o JPG; una foto
legible de un dibujo en papel sirve perfectamente) y se verán en el README.

---

## Qué tiene que mostrar el esquema de Q-Learning

El ciclo cerrado **estado → acción → recompensa → actualización → estado**,
tal como está implementado en `src/mountain_car/agents/qlearning.py`:

1. **Entorno (MountainCar)** entrega la observación continua `(posición, velocidad)`.
2. **Discretización** — `discretize()` convierte esa observación en una celda de
   la rejilla de 20 × 20 (la clave de la tabla).
3. **Tabla Q** — una fila por celda visitada, una columna por acción (3 acciones).
4. **Selección de acción** — `select_action()`, epsilon-greedy: con probabilidad
   `epsilon` una acción al azar, si no el `argmax` de la fila.
5. **El entorno devuelve** `recompensa = −1` y el siguiente estado.
6. **Actualización** — `_update()`:
   `Q(s,a) ← Q(s,a) + lr · [ r + γ · max Q(s′,·) − Q(s,a) ]`
   y la flecha vuelve de aquí a la tabla Q y al paso 1.
7. **Al final de cada episodio**, `epsilon` decae (1.0 → 0.01).

## Qué tiene que mostrar el esquema de DQN

Los tres elementos que la rúbrica nombra — **replay, red objetivo y
actualización de Bellman** — según `src/mountain_car/agents/dqn.py`:

1. **Entorno** entrega la observación continua (sin discretizar).
2. **Red online `q_net`** — MLP `2 → 128 → 128 → 3`; devuelve un Q-value por acción.
3. **Selección de acción** — epsilon-greedy **con exploración pegajosa**: al
   explorar, con probabilidad `stickiness = 0.9` repite la última acción
   exploratoria en vez de sortear una nueva. (Vale la pena marcarlo: es la
   corrección del Ejercicio 3.)
4. **Replay buffer** — se guarda la transición `(s, a, r, s′, terminated)` en una
   cola de 100 000; de ahí se saca un mini-lote aleatorio de 64.
5. **Red objetivo `target_net`** — una copia congelada de la red online que
   calcula `max Q_target(s′, ·)`. Dibuja la flecha de sincronización: se copian
   los pesos de `q_net` a `target_net` cada 10 episodios.
6. **Objetivo de Bellman** — `y = r + γ · max Q_target(s′,·) · (1 − terminated)`.
7. **Pérdida y paso de gradiente** — MSE entre `Q(s,a)` de la red online y `y`;
   el gradiente actualiza **solo la red online** (la flecha del gradiente no
   llega a la red objetivo — ese es el punto de tener dos redes).

---

> **Importante:** la rúbrica exige que estos dos esquemas sean **dibujo propio**
> y asigna **0 puntos** a un esquema "ausente o autogenerado por IA". Dibújalos
> tú (a mano, en Excalidraw, en PowerPoint, como prefieras) usando la lista de
> arriba como guion de qué cajas y flechas no pueden faltar.
