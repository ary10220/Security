"""Genera los videos sinteticos usados por los tests de preprocesamiento.

Se corre UNA sola vez; los .mp4 resultantes se commitean al repo. El objetivo
es que los tests no dependan de datasets externos (RWF, Real Life, Hockey)
y puedan correr en cualquier maquina / CI sin descargar nada.

Uso:
    python tests/fixtures/make_fixture.py

Produce tres archivos en tests/fixtures/:
    - sample_video.mp4   -> 90 frames (3 s @ 30 fps), 320x240, colores
      cambiando entre frames. Usado por los tests de forma y por el test
      de integracion del VideoPreprocessor.
    - short_video.mp4    -> 10 frames (<16), misma resolucion. Usado para
      verificar que extract_frames rellena hasta n_frames y no falla.
    - video_corrupto.mp4 -> bytes aleatorios con extension .mp4. Usado para
      verificar que extract_frames lanza ValueError cuando OpenCV no puede
      abrir el archivo.
"""

from pathlib import Path

import cv2
import numpy as np

# Resolucion deliberadamente baja: los tests no miden calidad visual, solo
# forma de los arreglos. Mas chico -> archivo mas liviano para el repo.
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # codec disponible por defecto en Windows

FIXTURES_DIR = Path(__file__).parent


def make_synthetic_video(output_path: Path, n_frames: int) -> None:
    """Genera un .mp4 con n_frames y colores que cambian frame a frame.

    Los colores varian con el indice (BGR), lo que ayuda a verificar
    manualmente que la secuencia extraida tiene variacion temporal (no es
    el mismo frame repetido). No se usa un patron fijo para los asserts:
    los tests solo chequean forma y rango, no pixeles exactos de este video.

    Args:
        output_path: ruta destino del .mp4.
        n_frames: cantidad de frames a escribir.
    """
    writer = cv2.VideoWriter(
        str(output_path), FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter no pudo abrir {output_path}")

    for i in range(n_frames):
        # Fondo gris con un canal variable; util para ver movimiento al
        # abrir el video a ojo durante debug.
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)
        # Variamos el canal rojo (indice 2 en BGR) con el numero de frame.
        frame[:, :, 2] = (i * 255 // max(n_frames - 1, 1))
        writer.write(frame)
    writer.release()


def make_corrupt_video(output_path: Path, size_bytes: int = 1024) -> None:
    """Escribe bytes aleatorios con extension .mp4 para simular archivo corrupto.

    OpenCV al intentar abrirlo devuelve isOpened() False, lo que dispara
    el ValueError que esperamos probar en extract_frames.

    Args:
        output_path: ruta destino del archivo basura.
        size_bytes: cuantos bytes aleatorios escribir (1 KB alcanza).
    """
    # np.random en vez de os.urandom para que el fixture sea reproducible
    # con una semilla fija: dos corridas producen el mismo archivo.
    rng = np.random.default_rng(seed=42)
    random_bytes = rng.integers(0, 256, size=size_bytes, dtype=np.uint8).tobytes()
    output_path.write_bytes(random_bytes)


def main() -> None:
    """Genera los tres fixtures en tests/fixtures/."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = FIXTURES_DIR / "sample_video.mp4"
    short_path = FIXTURES_DIR / "short_video.mp4"
    corrupt_path = FIXTURES_DIR / "video_corrupto.mp4"

    print(f"Generando {sample_path}  (90 frames)")
    make_synthetic_video(sample_path, n_frames=90)

    print(f"Generando {short_path}  (10 frames, mas corto que n_frames=16)")
    make_synthetic_video(short_path, n_frames=10)

    print(f"Generando {corrupt_path}  (bytes aleatorios)")
    make_corrupt_video(corrupt_path)

    print("Listo.")


if __name__ == "__main__":
    main()
