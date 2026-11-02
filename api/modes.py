"""Contrato de los modos de inferencia de Vigía."""

from __future__ import annotations

import json
from dataclasses import dataclass

from api import model_loader


@dataclass(frozen=True)
class Mode:
    """Identidad, artefacto y explicación conservadora de un clasificador."""

    label: str
    artifact: str
    reason: str


MODES = {
    "peleas": Mode("Peleas · RWF-2000", "mobilenet_lstm",
                   "Patrón visual similar a clips Fight de RWF-2000; requiere revisión humana."),
    "interior": Mode("Interior / joyería", "interior",
                     "Patrón visual similar a eventos observables de riesgo en interior; revisar el contexto."),
    "exterior": Mode("Exterior", "exterior",
                     "Patrón visual similar a eventos observables de riesgo en exterior; revisar el contexto."),
}


def get_mode(mode_id: str) -> Mode:
    """Obtiene un modo conocido sin suponer equivalencia entre clases."""
    if mode_id not in MODES:
        raise ValueError("Modo desconocido: usar peleas, interior o exterior.")
    return MODES[mode_id]


def mode_status(mode_id: str) -> dict:
    """Comprueba pesos y evaluación propia antes de anunciar disponibilidad."""
    mode = get_mode(mode_id)
    checkpoint = model_loader.checkpoint_path(mode.artifact)
    metrics_path = model_loader.ARTIFACTS_DIR / f"metrics_{mode.artifact}.json"
    if not checkpoint.is_file():
        detail = "Falta el checkpoint entrenado."
    elif not metrics_path.is_file():
        detail = "Faltan métricas de prueba del modo."
    else:
        try:
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            if mode_id != "peleas" and report.get("mode") != mode_id:
                raise ValueError("Las métricas pertenecen a otro modo.")
            if not isinstance(report.get("final", {}).get("confusion_matrix"), list):
                raise ValueError("Falta la matriz de confusión de prueba.")
            loaded = model_loader.load_model(
                mode.artifact,
                expected_mode=mode_id if mode_id != "peleas" else None,
            )
            detail = "Disponible" if loaded.from_checkpoint else "Checkpoint inválido."
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            detail = f"Evaluación inválida: {exc}"
    return {
        "id": mode_id,
        "label": mode.label,
        "available": detail == "Disponible",
        "detail": detail,
        "reason": mode.reason,
        "metrics": mode.artifact if detail == "Disponible" else None,
    }
