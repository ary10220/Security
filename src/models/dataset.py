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
import re
from collections import Counter
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

# Los clips de RWF-2000 se llaman <fuente>_<n>.avi: todos los que comparten
# <fuente> son tramos del mismo video original (ver data/README.md).
SOURCE_PATTERN = re.compile(r"^(.+)_\d+$")

# RWF-2000 repite el split en el nombre de la carpeta de clase (Train_Fight,
# Val_NonFight). Ese prefijo no es parte de la clase: sin quitarlo, las
# metricas y el frontend mostrarian "Train_Fight" aunque se evalue en val/.
SPLIT_PREFIX = re.compile(r"^(train|val|test)_", re.IGNORECASE)

logger = logging.getLogger(__name__)


def class_display_name(folder_name: str) -> str:
    """Nombre de la clase sin el prefijo de split de la carpeta.

    Args:
        folder_name: nombre de la carpeta de clase, p.ej. `Train_Fight`.

    Returns:
        `Fight` para `Train_Fight` o `Val_Fight`. Si no hay prefijo de split
        (`Violence`, `NonFight`), el nombre queda igual.
    """
    return SPLIT_PREFIX.sub("", folder_name)


def source_id(video_path: str | Path) -> str:
    """Devuelve el video fuente de un clip, segun su nombre.

    Args:
        video_path: ruta al clip, p.ej. `.../Train_Fight/abc123_0.avi`.

    Returns:
        El prefijo antes del ultimo `_<numero>` (`abc123`). Si el nombre no
        sigue ese patron, el nombre completo sin extension: el clip queda
        como su propia fuente.
    """
    stem = Path(video_path).stem
    match = SOURCE_PATTERN.match(stem)
    return match.group(1) if match else stem


def split_by_source(
    samples: list[tuple[Path, int]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Separa un val interno dividiendo por video fuente, estratificado.

    El EDA mostro que los 1600 clips de train de RWF-2000 salen de ~961
    videos fuente. Si se divide por clip, tramos casi iguales del mismo video
    caen en train y en val, y el F1 de val (el que elige el checkpoint) sale
    inflado. Por eso se reparten fuentes enteras, y por clase, para que val
    tenga la misma proporcion Fight/NonFight que train.

    Si una clase tiene una sola fuente (p.ej. Real Life, donde el nombre
    `V_398` no dice de que video sale, o el mini dataset), en esa clase se
    divide por clip y se avisa: no hay forma de dividir por fuente.

    Args:
        samples: pares (ruta, label) de `VideoDataset.samples`.
        val_fraction: proporcion de fuentes (o clips) que van a val.
        seed: semilla para que el split sea el mismo en cada corrida.

    Returns:
        Tupla `(indices_train, indices_val)` sobre `samples`, ordenados.
    """
    rng = random.Random(seed)

    groups: dict[str, list[int]] = {}
    for i, (path, _) in enumerate(samples):
        groups.setdefault(source_id(path), []).append(i)

    # Cada fuente se asigna a la clase mayoritaria de sus clips. En RWF-2000
    # una fuente es de una sola clase; si no lo fuera, igual queda entera en
    # un lado del split, que es lo que evita la fuga.
    by_label: dict[int, list[str]] = {}
    for source, indices in sorted(groups.items()):
        label = Counter(samples[i][1] for i in indices).most_common(1)[0][0]
        by_label.setdefault(label, []).append(source)

    val: list[int] = []
    for label, sources in sorted(by_label.items()):
        if len(sources) >= 2:
            rng.shuffle(sources)
            n_val = max(1, round(len(sources) * val_fraction))
            for source in sources[:n_val]:
                val.extend(groups[source])
        else:
            clips = list(groups[sources[0]])
            logger.warning(
                "La clase %d tiene una sola fuente: se divide por clip.", label
            )
            rng.shuffle(clips)
            n_val = max(1, round(len(clips) * val_fraction)) if len(clips) >= 2 else 0
            val.extend(clips[:n_val])

    val_set = set(val)
    train = [i for i in range(len(samples)) if i not in val_set]
    return train, sorted(val_set)


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
        # imprima nombres en vez de 0/1. Sin el prefijo de split, asi train/
        # y val/ dan los mismos nombres (Fight, NonFight).
        self.classes = [
            class_display_name(name)
            for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])
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
