"""Genera un mini-dataset sintetico para smoke tests de entrenamiento.

Produce 16 videos en `data/raw/mini/`, con la estructura que espera
`VideoDataset`:

    data/raw/mini/
      train/
        Fight/     fight_01..06.mp4   (6 clips)
        NonFight/  non_01..06.mp4     (6 clips)
      val/
        Fight/     fight_val_01..02.mp4   (2 clips)
        NonFight/  non_val_01..02.mp4     (2 clips)

Los videos NO tienen senal diferencial real: las metricas de un smoke test
van a ser aleatorias. El objetivo es verificar que el pipeline (DataLoader,
forward, backward, guardado de checkpoint) corra fin a fin, no entrenar un
modelo util. El entrenamiento serio es con RWF-2000.

La carpeta `data/raw/mini/` esta en `.gitignore` (`data/raw/`): solo se
versiona este script, no los `.mp4` generados.

Uso:
    python scripts/make_mini_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30
N_FRAMES_PER_VIDEO = 90  # 3 segundos
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# Raiz del repo: subimos dos niveles desde scripts/make_mini_dataset.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "mini"


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    """Escribe la secuencia de frames como .mp4 con mp4v.

    Args:
        path: ruta destino.
        frames: lista de frames BGR uint8 de la misma resolucion.
    """
    writer = cv2.VideoWriter(
        str(path), FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter no pudo abrir {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _make_fight_frames(seed: int) -> list[np.ndarray]:
    """Clip con transiciones bruscas de color.

    Cada ~5 frames saltamos a un color base nuevo elegido al azar. Es la
    forma sintetica de simular "movimiento rapido": el LSTM ve cambios
    grandes de frame a frame. No pretende parecer pelea real.
    """
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    base_color = rng.integers(0, 256, size=3, dtype=np.uint8)
    for i in range(N_FRAMES_PER_VIDEO):
        if i % 5 == 0:
            # Salto de color: nuevo base elegido al azar.
            base_color = rng.integers(0, 256, size=3, dtype=np.uint8)
        # Ruido chico por pixel para que no sea un color plano puro.
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), base_color, dtype=np.uint8)
        noise = rng.integers(-15, 16, size=frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _make_nonfight_frames(seed: int) -> list[np.ndarray]:
    """Clip con transiciones suaves de color.

    Interpolamos linealmente entre dos colores a lo largo del clip. El
    cambio por frame es minimo, para que el LSTM lo lea como "escena
    tranquila" distinta del clip de pelea.
    """
    rng = np.random.default_rng(seed)
    color_a = rng.integers(0, 256, size=3).astype(np.float32)
    color_b = rng.integers(0, 256, size=3).astype(np.float32)
    frames: list[np.ndarray] = []
    for i in range(N_FRAMES_PER_VIDEO):
        t = i / max(N_FRAMES_PER_VIDEO - 1, 1)  # 0 -> 1 a lo largo del clip
        base_color = (1 - t) * color_a + t * color_b
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), base_color.astype(np.uint8),
                        dtype=np.uint8)
        noise = rng.integers(-5, 6, size=frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _generate_batch(output_dir: Path, prefix: str, count: int,
                    seed_offset: int, kind: str) -> None:
    """Genera `count` videos del tipo pedido en `output_dir`.

    Args:
        output_dir: carpeta donde guardar los .mp4 (se crea si falta).
        prefix: nombre base de los archivos, p.ej. "fight".
        count: cuantos videos generar.
        seed_offset: base para las semillas, para que no se repitan entre
            lotes y las corridas sean reproducibles.
        kind: "fight" o "nonfight", elige el generador.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    make = _make_fight_frames if kind == "fight" else _make_nonfight_frames
    for i in range(1, count + 1):
        frames = make(seed=seed_offset + i)
        path = output_dir / f"{prefix}_{i:02d}.mp4"
        _write_video(path, frames)
        print(f"  escrito {path.relative_to(REPO_ROOT)}")


def main() -> None:
    """Genera los 16 videos del mini-dataset."""
    print(f"Mini-dataset en {OUTPUT_ROOT}")
    _generate_batch(OUTPUT_ROOT / "train" / "Fight", "fight", 6,
                    seed_offset=100, kind="fight")
    _generate_batch(OUTPUT_ROOT / "train" / "NonFight", "non", 6,
                    seed_offset=200, kind="nonfight")
    _generate_batch(OUTPUT_ROOT / "val" / "Fight", "fight_val", 2,
                    seed_offset=300, kind="fight")
    _generate_batch(OUTPUT_ROOT / "val" / "NonFight", "non_val", 2,
                    seed_offset=400, kind="nonfight")
    print("Listo.")


if __name__ == "__main__":
    main()
