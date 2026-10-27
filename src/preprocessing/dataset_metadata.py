"""Metadatos de los videos de un dataset (insumo del EDA).

Recorre una carpeta de videos y arma una tabla con duracion, fps,
resolucion, tamano en disco y clase de cada clip. Solo lee las cabeceras
con OpenCV, no decodifica el video completo, asi que escanear 2000 clips
toma segundos.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def label_from_path(video_path: Path) -> str:
    """Deduce la clase de un clip a partir del nombre de su carpeta.

    Los tres datasets nombran la carpeta con la clase (`Violence`,
    `NonViolence`, `Train_Fight`, `Val_NonFight`...). Se evalua primero el
    prefijo "non" porque "NonViolence" tambien contiene "violence".

    Args:
        video_path: ruta al video.

    Returns:
        "violencia" o "no_violencia".

    Raises:
        ValueError: si ninguna carpeta del camino indica la clase.
    """
    for part in reversed(video_path.parts[:-1]):
        name = part.lower()
        if "nonviolence" in name or "nonfight" in name:
            return "no_violencia"
        if "violence" in name or "fight" in name:
            return "violencia"
    raise ValueError(f"No se pudo deducir la clase de: {video_path}")


def split_from_path(video_path: Path) -> str:
    """Devuelve el split ("train", "val") si el camino lo indica, o "".

    RWF-2000 viene ya dividido en `train/` y `val/`; Real Life no.

    Args:
        video_path: ruta al video.

    Returns:
        "train", "val" o cadena vacia si el dataset no trae split.
    """
    parts = [p.lower() for p in video_path.parts[:-1]]
    for split in ("train", "val"):
        if split in parts:
            return split
    return ""


def read_video_info(video_path: Path) -> dict:
    """Lee los metadatos de un video sin decodificarlo completo.

    Args:
        video_path: ruta al video.

    Returns:
        Diccionario con bytes, frames, fps, ancho, alto, segundos y `legible`
        (True si OpenCV logra leer el primer frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "bytes": video_path.stat().st_size,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        # Leer el primer frame detecta archivos que abren pero estan rotos.
        "legible": bool(cap.isOpened() and cap.read()[0]),
    }
    cap.release()
    # Si el fps viene en 0 (metadata rota) se evita dividir por cero.
    info["segundos"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def threshold_shortcut(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    """Mide cuanto separa las clases una sola caracteristica con un umbral.

    Busca en `train` el umbral (y el sentido: mayor o menor) que mejor
    separa las clases y luego lo evalua en `test`. Si un dato que no es el
    comportamiento (fps, resolucion, bitrate) separa bien las clases, el
    modelo podria aprenderlo como atajo. Se evalua en datos distintos a los
    usados para elegir el umbral para no sobreestimar la exactitud.

    Args:
        train_x: valores de la caracteristica para elegir el umbral.
        train_y: etiquetas binarias (1 = violencia) de `train_x`.
        test_x: valores de la caracteristica para evaluar.
        test_y: etiquetas binarias de `test_x`.

    Returns:
        Diccionario con `umbral`, `sentido` (">" o "<=": la regla predice
        violencia si x cumple la condicion), `acc_train` y `acc_test`.
    """
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=int)
    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y, dtype=int)

    # Se prueban como umbrales los puntos medios entre valores distintos
    # consecutivos; basta con eso para cubrir todas las particiones posibles.
    values = np.unique(train_x)
    candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values

    best = {"umbral": float(values[0]), "sentido": ">", "acc_train": 0.0}
    for thr in candidates:
        for direction in (">", "<="):
            pred = (train_x > thr) if direction == ">" else (train_x <= thr)
            acc = float((pred.astype(int) == train_y).mean())
            if acc > best["acc_train"]:
                best = {"umbral": float(thr), "sentido": direction, "acc_train": acc}

    pred_test = (test_x > best["umbral"]) if best["sentido"] == ">" else (test_x <= best["umbral"])
    best["acc_test"] = float((pred_test.astype(int) == test_y).mean())
    return best


def scan_dataset(root: str | Path, dataset: str) -> pd.DataFrame:
    """Escanea recursivamente una carpeta y arma la tabla de metadatos.

    Args:
        root: carpeta raiz del dataset (por ejemplo `data/raw/real_life`).
        dataset: nombre que se guarda en la columna `dataset`.

    Returns:
        DataFrame con una fila por video y las columnas: dataset, split,
        clase, archivo, formato, ruta, bytes, frames, fps, ancho, alto,
        segundos y legible. Vacio si no hay videos.
    """
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "split": split_from_path(path.relative_to(root)),
                "clase": label_from_path(path.relative_to(root)),
                "archivo": path.name,
                "formato": path.suffix.lower().lstrip("."),
                "ruta": str(path),
                **read_video_info(path),
            }
        )
    return pd.DataFrame(rows)
