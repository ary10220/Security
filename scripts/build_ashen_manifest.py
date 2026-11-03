"""Arma el CSV de etiquetas de los modos interior/exterior a partir de ASHEN.

Entrada:
    - data/ashen/videos.csv: una fila por video con clase (Robbery, Assault o
      Normal), ambiente (interior, exterior o excluido) y split (train/test).
      Es la fuente de verdad de las decisiones humanas.
    - data/ashen/anotaciones_temporales_ashen.txt: intervalos del evento, en
      fotogramas, de los videos de Robbery y Assault (archivo oficial de ASHEN).
    - La carpeta con los videos descargados (anomaly/ y normal/).

Salida: el CSV que consume scripts/prepare_risk_data.py (una fila por clip).

Uso (desde la raiz del repo):
    python -m scripts.build_ashen_manifest --videos-root /ruta/a/ashen \
        --output /ruta/a/ashen/risk_labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

# Clips de 5 s: la misma duracion que los de RWF-2000, con los que se
# disenaron el muestreo de 16 frames y la ventana de la API.
WINDOW_S = 5.0
# Tope de clips por video. Sin tope, un robo largo (hay eventos de 5 min)
# aportaria 60 clips casi iguales y dominaria el entrenamiento.
MAX_RISK_CLIPS = 6
MAX_NORMAL_CLIPS = 3
# Un evento mas corto que esto no deja ver nada util en 16 frames.
MIN_EVENT_S = 2.0
RIGHTS = "ASHEN (UCF-Crime) CC BY-NC 4.0, uso academico"
FIELDS = ["file", "source_id", "mode", "label", "start_sec", "end_sec", "split", "rights", "notes"]


def read_annotations(path: Path) -> dict[str, list[int]]:
    """Lee el archivo oficial: video, clase, inicio1, fin1, inicio2, fin2 (fotogramas).

    Args:
        path: archivo de anotaciones temporales de ASHEN.

    Returns:
        Nombre del video -> [inicio1, fin1, inicio2, fin2]; -1 si no hay segundo evento.
    """
    annotations = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 6:
            annotations[parts[0]] = [int(value) for value in parts[2:6]]
    return annotations


def video_timing(path: Path) -> tuple[float, float]:
    """Fps y duracion del video, leidos con OpenCV como en el resto del pipeline.

    Args:
        path: ruta del video.

    Returns:
        (fps redondeado a 2 decimales, duracion en segundos redondeada a 0,1).
    """
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"No se pudo leer el fps de {path}")
    return round(fps, 2), round(frames / fps, 1)


def risk_windows(annotation: list[int], fps: float, duration: float) -> list[tuple[float, float]]:
    """Ventanas de 5 s dentro de los intervalos anotados del evento.

    Las anotaciones vienen en fotogramas; ASHEN mezcla videos de 25 y 30 fps,
    por eso se convierte con el fps de cada video.

    Args:
        annotation: [inicio1, fin1, inicio2, fin2] en fotogramas.
        fps: fotogramas por segundo del video.
        duration: duracion del video en segundos.

    Returns:
        Intervalos (inicio, fin) en segundos, como maximo MAX_RISK_CLIPS.
    """
    windows: list[tuple[float, float]] = []
    for start, end in ((annotation[0], annotation[1]), (annotation[2], annotation[3])):
        if start < 0:
            continue
        # 0,2 s de margen al final: el ultimo fotograma no siempre se decodifica.
        start_s, end_s = start / fps, min(end / fps, duration - 0.2)
        if end_s - start_s < WINDOW_S:
            # Evento corto: un solo clip que cubre todo el evento.
            if end_s - start_s >= MIN_EVENT_S:
                windows.append((start_s, end_s))
            continue
        t = start_s
        while t + WINDOW_S <= end_s:
            windows.append((t, t + WINDOW_S))
            t += WINDOW_S
    if len(windows) > MAX_RISK_CLIPS:
        # Repartidas a lo largo del evento, no solo las primeras.
        step = len(windows) / MAX_RISK_CLIPS
        windows = [windows[int(i * step)] for i in range(MAX_RISK_CLIPS)]
    return windows


def normal_windows(duration: float) -> list[tuple[float, float]]:
    """Hasta MAX_NORMAL_CLIPS ventanas de 5 s repartidas en un video normal.

    Args:
        duration: duracion del video en segundos.

    Returns:
        Intervalos (inicio, fin) en segundos.
    """
    usable = duration - 1.0  # medio segundo de margen al inicio y al final
    count = min(MAX_NORMAL_CLIPS, int(usable // WINDOW_S))
    return [(0.5 + i * usable / count, 0.5 + i * usable / count + WINDOW_S)
            for i in range(count)]


def build_rows(videos_csv: Path, annotations: dict[str, list[int]],
               videos_root: Path) -> list[dict]:
    """Genera una fila por clip para cada video no excluido.

    Args:
        videos_csv: data/ashen/videos.csv.
        annotations: salida de read_annotations.
        videos_root: carpeta con anomaly/ y normal/ descargados.

    Returns:
        Filas en el formato de scripts/prepare_risk_data.py.
    """
    rows = []
    with videos_csv.open(encoding="utf-8", newline="") as handle:
        for video in csv.DictReader(handle):
            if video["ambiente"] == "excluido":
                continue
            fps, duration = video_timing(videos_root / video["ruta_hf"])
            stem = Path(video["video"]).stem
            if video["clase"] == "Normal":
                label, windows = "Normal", normal_windows(duration)
                notes = f"Normal; ambiente {video['ambiente']}"
            else:
                label = "Riesgo"
                windows = risk_windows(annotations[video["video"]], fps, duration)
                notes = f"{video['clase']}; ambiente {video['ambiente']}; intervalo de anotacion ASHEN"
            # En prepare_risk_data el test se llama "val" (la carpeta val/).
            split = "val" if video["split"] == "test" else "train"
            for start, end in windows:
                rows.append({"file": video["ruta_hf"], "source_id": stem,
                             "mode": video["ambiente"], "label": label,
                             "start_sec": round(start, 2), "end_sec": round(end, 2),
                             "split": split, "rights": RIGHTS, "notes": notes})
    return rows


def main() -> int:
    """Escribe el CSV de etiquetas listo para prepare_risk_data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, required=True,
                        help="Carpeta con anomaly/ y normal/ descargados de ASHEN.")
    parser.add_argument("--videos-csv", type=Path, default=Path("data/ashen/videos.csv"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/ashen/anotaciones_temporales_ashen.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.videos_csv, read_annotations(args.annotations), args.videos_root)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} clips en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
