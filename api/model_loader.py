"""Carga del modelo entrenado para la API, una sola vez por proceso.

La API no entrena ni reconstruye la arquitectura a mano: usa
`load_checkpoint` de src/models/model.py, el mismo que usa train.py para
evaluar el test. El checkpoint es artifacts/model_<nombre>.pt, como lo guarda
train.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from src.models.model import build_model, load_checkpoint

logger = logging.getLogger(__name__)

# Carpeta de checkpoints y metricas. Se puede cambiar con la variable de
# entorno ARTIFACTS_DIR (los tests apuntan a una carpeta temporal).
ARTIFACTS_DIR = Path(os.environ.get(
    "ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"
))

# Modelo que carga la API por defecto: el que gano la comparacion del plan
# (seccion 4). Se cambia con MODEL_NAME=resnet_lstm, por ejemplo.
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "mobilenet_lstm")


@dataclass(frozen=True)
class LoadedModel:
    """Modelo listo para predecir y de donde salieron sus pesos.

    Attributes:
        model: red en modo eval, en `device`.
        name: nombre del modelo (uno de MODEL_NAMES).
        device: "cpu" o "cuda".
        from_checkpoint: False si no habia checkpoint y se usan pesos random.
    """

    model: nn.Module
    name: str
    device: str
    from_checkpoint: bool


def checkpoint_path(name: str, artifacts_dir: Path | None = None) -> Path:
    """Ruta del checkpoint de un modelo, con el nombre que usa train.py."""
    return Path(artifacts_dir or ARTIFACTS_DIR) / f"model_{name}.pt"


@lru_cache(maxsize=None)
def load_model(name: str = DEFAULT_MODEL,
               artifacts_dir: Path | None = None) -> LoadedModel:
    """Devuelve el modelo `name`, cargandolo solo la primera vez.

    lru_cache hace de singleton: cada request reutiliza la misma instancia en
    vez de leer el .pt de disco (17-95 MB) en cada prediccion.

    Si no existe el checkpoint, se avisa y se usan pesos random: asi la API y
    el frontend se pueden desarrollar antes de tener un modelo entrenado.
    /health lo informa con model_loaded=false, para que nadie confunda esas
    predicciones con resultados reales.

    Args:
        name: uno de MODEL_NAMES.
        artifacts_dir: carpeta del checkpoint; None usa ARTIFACTS_DIR.

    Returns:
        LoadedModel con el modelo en modo eval.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = checkpoint_path(name, artifacts_dir)
    if path.is_file():
        logger.info("Cargando checkpoint %s en %s", path, device)
        try:
            return LoadedModel(load_checkpoint(path, device=device), name, device, True)
        except (KeyError, RuntimeError) as exc:
            # Un .pt de antes de save_checkpoint (sin "model_name") o de otra
            # arquitectura no debe tirar la API: se avisa y se sigue sin el.
            logger.warning("No se pudo cargar %s (%r).", path, exc)

    logger.warning(
        "Sin checkpoint valido en %s: se usan pesos RANDOM. Las predicciones "
        "no significan nada hasta entrenar con train.py.", path,
    )
    # pretrained=False: sin checkpoint no vale la pena descargar ImageNet,
    # y asi la API arranca sin internet.
    model = build_model(name, pretrained=False).to(device).eval()
    return LoadedModel(model, name, device, False)
