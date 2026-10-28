"""Dataset de video para el entrenamiento (fase 3 del plan).

Carga los videos de una carpeta organizada por clase y los pasa por
`VideoPreprocessor` para producir tensores `(T, C, H, W)` que los modelos
de `src/models/model.py` consumen directamente.

Estructura esperada del arbol de archivos:

    root_dir/
      <split>/           # p.ej. "train" o "val" (opcional)
        <clase_a>/
          video1.mp4
          video2.avi
        <clase_b>/
          ...

Si `split=""` o no existe `root_dir/<split>/`, se buscan las clases en
`root_dir/` directamente (p.ej. Real Life no trae split, hay que armarlo
afuera).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset

from src.preprocessing import VideoPreprocessor

# Extensiones aceptadas. OpenCV abre varias mas (mov, mkv, webm) pero el
# proyecto solo trabaja con mp4 y avi; dejar la lista corta evita incluir
# por error archivos de audio o imagenes sueltas.
VIDEO_EXTENSIONS = (".mp4", ".avi")

# Palabras clave que indican "clase sospechosa" en el nombre de carpeta.
# Si la carpeta se llama "Fight", "Train_Fight" o "Violence", es sospechosa
# (label 1). Si no, es normal (label 0). Siempre se puede sobreescribir
# con `label_map` para datasets con otra nomenclatura.
SUSPICIOUS_KEYWORDS = ("fight", "violence")

# Marcadores de la clase NORMAL: carpetas como "NonFight", "NonViolence",
# "Val_NonFight". "non" por si solo basta: queda fuera cualquier palabra
# que no sea negacion (vigilance, movement, etc. no empiezan por "non").
NEGATIVE_MARKERS = ("non",)

# Cuantas veces reintenta `__getitem__` antes de rendirse si todos los videos
# vecinos fallan. 10 alcanza para saltear archivos corruptos sueltos sin
# meterse en un bucle si el dataset entero esta roto.
MAX_RETRIES = 10

logger = logging.getLogger(__name__)


def _detect_label_map(class_dirs: list[Path]) -> dict[str, int]:
    """Mapea nombres de carpeta a etiquetas 0 (normal) y 1 (sospechoso).

    Args:
        class_dirs: subdirectorios ordenados que representan clases.

    Returns:
        Diccionario `nombre_carpeta -> label`.

    Raises:
        ValueError: si no se detectan exactamente dos clases o si no se
            puede identificar cual es la sospechosa por sus nombres.
    """
    # El modelo es binario (ver NUM_CLASSES en model.py): exigir exactamente
    # dos subcarpetas evita silenciar bugs si se agrega una tercera por error.
    if len(class_dirs) != 2:
        raise ValueError(
            f"Se esperaban 2 subcarpetas de clase, se encontraron "
            f"{len(class_dirs)}: {[d.name for d in class_dirs]}"
        )

    labels: dict[str, int] = {}
    for class_dir in class_dirs:
        name_lower = class_dir.name.lower()
        has_keyword = any(k in name_lower for k in SUSPICIOUS_KEYWORDS)
        # "nonfight"/"nonviolence" contienen la keyword pero son la clase
        # normal: si aparece un marcador negativo, cancela el match.
        is_negative = any(m in name_lower for m in NEGATIVE_MARKERS)
        is_suspicious = has_keyword and not is_negative
        labels[class_dir.name] = 1 if is_suspicious else 0

    # Si las dos carpetas quedaron con la misma etiqueta, es que ninguna
    # tiene las palabras clave (o las tienen las dos). Pedimos un mapping
    # explicito en vez de adivinar: equivocarse en la etiqueta arruina todo.
    if set(labels.values()) != {0, 1}:
        raise ValueError(
            "No se pudo autodetectar la clase sospechosa en "
            f"{list(labels)}. Pasa `label_map=...` al construir el dataset."
        )
    return labels


class VideoDataset(Dataset):
    """Dataset de videos clasificados en carpetas por clase.

    Attributes:
        samples: lista de pares (ruta_al_video, label) construida al iniciar.
        preprocessor: pipeline que convierte el video en tensor.
        classes: nombres de las clases ordenados por su etiqueta (0, 1).
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        preprocessor: VideoPreprocessor | None = None,
        label_map: Mapping[str, int] | None = None,
    ) -> None:
        """Escanea `root_dir` y arma la lista de muestras.

        Args:
            root_dir: carpeta raiz del dataset. Puede contener un `split/`
                adentro o directamente las carpetas de clase.
            split: subcarpeta de split a mirar (p.ej. "train", "val"). Si es
                "" o no existe, se usan las clases directas de `root_dir`.
            preprocessor: pipeline a aplicar a cada video. Si es None, se
                crea uno nuevo con `train=(split == "train")`.
            label_map: mapeo explicito `nombre_carpeta -> label`. Util para
                datasets con nombres que no contienen "fight"/"violence".
        """
        root = Path(root_dir)
        # Primero probamos con split; si no existe, caemos al root plano.
        split_dir = root / split if split else root
        if not split_dir.is_dir():
            split_dir = root
        self.split_dir = split_dir

        # Ordenar los subdirectorios da un mapeo estable entre corridas:
        # si Python los devolviera en otro orden, cambiarian las etiquetas.
        class_dirs = sorted(
            [d for d in split_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if label_map is None:
            label_map = _detect_label_map(class_dirs)
        else:
            # Verificar que el label_map cubra todas las carpetas, para que
            # un typo del usuario falle con mensaje claro y no en silencio.
            missing = [d.name for d in class_dirs if d.name not in label_map]
            if missing:
                raise ValueError(
                    f"`label_map` no incluye las carpetas: {missing}"
                )

        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = label_map[class_dir.name]
            for ext in VIDEO_EXTENSIONS:
                for video_path in sorted(class_dir.glob(f"*{ext}")):
                    self.samples.append((video_path, label))

        if not self.samples:
            raise ValueError(
                f"No se encontraron videos en {split_dir} con extensiones "
                f"{VIDEO_EXTENSIONS}"
            )

        # Guardamos clases ordenadas por label para que el reporte de metricas
        # imprima nombres en vez de 0/1.
        self.classes = [
            name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])
        ]

        self.preprocessor = preprocessor or VideoPreprocessor(
            train=(split == "train")
        )

    def __len__(self) -> int:
        """Cantidad de videos."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Devuelve `(tensor, label)` o salta videos corruptos.

        Si el video en `idx` no se puede procesar (archivo roto, codec no
        soportado), se loguea un warning y se intenta con un vecino aleatorio.
        El entrenamiento no debe caerse por un `.mp4` suelto; en 2000+ videos
        es razonable que 1 o 2 esten rotos.

        Args:
            idx: indice de la muestra.

        Returns:
            Tupla `(tensor float32 de forma (T, C, H, W), label int)`.

        Raises:
            RuntimeError: si tras MAX_RETRIES intentos no se logra procesar
                ningun video (sintoma de que el dataset entero esta roto).
        """
        tried = set()
        current = idx
        for _ in range(MAX_RETRIES):
            tried.add(current)
            video_path, label = self.samples[current]
            try:
                tensor = self.preprocessor(str(video_path))
                return tensor, label
            except Exception as exc:
                logger.warning(
                    "Fallo al procesar %s (%s); se intenta otro video.",
                    video_path, exc,
                )
                # Elegimos un vecino al azar para no caer siempre en el
                # mismo video roto si el siguiente tambien lo esta.
                remaining = [i for i in range(len(self.samples)) if i not in tried]
                if not remaining:
                    break
                current = random.choice(remaining)

        raise RuntimeError(
            f"No se pudo procesar ningun video tras {MAX_RETRIES} intentos "
            f"desde idx={idx}. Revisar la integridad del dataset."
        )
