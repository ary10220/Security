"""Modelos Pydantic de las respuestas de la API.

Ademas de los campos del plan (seccion 5), /predict devuelve los que pide el
contrato del frontend (frontend/README.md): asi el frontend de Diego y
Grageda funciona sin cambios y el plan se cumple igual.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Estado de la API y del modelo cargado."""

    status: Literal["ok"]
    # False = no habia checkpoint y se usan pesos random (ver model_loader).
    model_loaded: bool
    device: Literal["cpu", "cuda"]
    model: str


class MetricsResponse(BaseModel):
    """Contenido de artifacts/metrics_<modelo>.json tal cual lo escribe train.py.

    Solo se fijan los campos que siempre estan; el resto (pr_curve, history,
    inference_ms...) pasa sin tocar, para no tener que cambiar la API cada
    vez que train.py guarde una metrica nueva.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    final: dict


class TimelineSegment(BaseModel):
    """Un tramo del video y la probabilidad de que sea sospechoso."""

    start: float
    end: float
    # Probabilidad (softmax) de la clase sospechoso, no la etiqueta: el
    # frontend aplica el umbral con el slider sin volver a llamar a la API.
    score: float


class VideoInfo(BaseModel):
    """Datos del video subido (campo `video` del contrato del frontend)."""

    name: str
    duration_s: float


class PredictResponse(BaseModel):
    """Resultado de /predict y /predict/clip."""

    verdict: Literal["suspicious", "normal"]
    confidence: float
    timeline: list[TimelineSegment]
    video_duration: float
    # Id para pedir los frames del preprocesamiento a /preprocessing_demo.
    video_id: str
    threshold: float
    # Campos del contrato del frontend: `segments` es el mismo timeline.
    model: str
    segment_seconds: float
    default_threshold: float
    segments: list[TimelineSegment]
    video: VideoInfo
    # Descarga del clip mas critico: todavia no implementada.
    clip_url: str | None = None


class PreprocessedFrame(BaseModel):
    """Un frame despues de extraerlo y redimensionarlo, como imagen."""

    index: int
    time: float
    src: str  # data:image/jpeg;base64,...


class PreprocessingDemoResponse(BaseModel):
    """Los 16 frames que ve el modelo, para la vista de preprocesamiento."""

    video_id: str
    n_frames: int
    size: list[int]
    frames: list[PreprocessedFrame]
