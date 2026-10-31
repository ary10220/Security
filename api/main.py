"""API FastAPI: endpoints de la seccion 5 del plan.

Levantar con:
    uvicorn api.main:app --reload
y abrir http://localhost:8000/docs para probar cada endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import model_loader
from api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    PreprocessedFrame,
    PreprocessingDemoResponse,
    TimelineSegment,
    VideoInfo,
)
from src.preprocessing import (
    IMAGE_SIZE,
    N_FRAMES,
    VideoPreprocessor,
    extract_frames,
    resize_and_convert,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Validaciones de entrada (seccion 5 del plan).
ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB, igual que MAX_UPLOAD_MB del frontend
MAX_DURATION_SECONDS = 120.0  # 2 minutos

# El video se analiza por tramos de 2 s (seccion 1 del plan). PROVISIONAL: el
# CLAUDE.md deja abierta la decision entre 2 s y 3 s.
SEGMENT_SECONDS = 2.0
# Ventana temporal contra falsos positivos (seccion 1): solo hay alerta si la
# sospecha dura mas de 3 s seguidos. Un empujon aislado no dispara.
MIN_ALERT_SECONDS = 3.0
DEFAULT_THRESHOLD = 0.5
# Tramos por forward: 8 x 16 frames es el mismo lote que en entrenamiento.
INFERENCE_BATCH = 8
UPLOAD_CHUNK = 1024 * 1024

# Frames de los ultimos videos analizados, por video_id, para
# /preprocessing_demo. En memoria y con tope: es una demo para la vista del
# frontend, no un almacenamiento; al reiniciar la API se pierden.
DEMO_CACHE_SIZE = 32
demo_cache: OrderedDict[str, list[PreprocessedFrame]] = OrderedDict()

# Mismo pipeline que en entrenamiento, sin augmentation (regla del CLAUDE.md:
# la API no reimplementa el preprocesamiento).
preprocessor = VideoPreprocessor(train=False)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Carga el modelo al arrancar, no en la primera prediccion.

    Asi el primer usuario no espera la carga y, si el checkpoint falta, el
    aviso aparece en la consola apenas se levanta uvicorn.
    """
    model_loader.load_model()
    yield


app = FastAPI(
    title="Deteccion de comportamiento sospechoso",
    description="Clasifica videos de camaras de seguridad en normal o sospechoso.",
    lifespan=lifespan,
)

# El frontend (Vite) corre en otro puerto: sin CORS el navegador bloquea
# las llamadas a la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Responde 400 en vez del 422 de FastAPI: el plan pide 400 para entradas invalidas."""
    detail = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
        for err in exc.errors()
    )
    return JSONResponse(status_code=400, content={"detail": detail})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Indica si la API responde y si hay un modelo entrenado cargado."""
    loaded = model_loader.load_model()
    return HealthResponse(
        status="ok",
        model_loaded=loaded.from_checkpoint,
        device=loaded.device,
        model=loaded.name,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> dict:
    """Devuelve las metricas de test del modelo cargado.

    Lee artifacts/metrics_<modelo>.json, el que escribe train.py. No las
    recalcula: evaluar el test set en cada request tardaria minutos.
    """
    name = model_loader.load_model().name
    path = model_loader.ARTIFACTS_DIR / f"metrics_{name}.json"
    if not path.is_file():
        raise HTTPException(
            404, f"Todavia no hay metricas de {name}: entrenar con train.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))


async def _save_upload(file: UploadFile, folder: Path) -> tuple[Path, str]:
    """Guarda el video en disco (OpenCV necesita una ruta) y calcula su hash.

    Se copia por partes y se corta apenas pasa de MAX_UPLOAD_BYTES, para no
    llenar el disco con un archivo que igual se va a rechazar.

    Returns:
        Tupla (ruta del archivo, sha256 en hexadecimal).

    Raises:
        HTTPException: 400 si esta vacio, 413 si pasa de 100 MB.
    """
    path = folder / f"video{Path(file.filename).suffix.lower()}"
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Video demasiado grande: el maximo es 100 MB.")
            digest.update(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "El video esta vacio.")
    return path, digest.hexdigest()


def _video_duration(path: Path) -> float:
    """Duracion en segundos, o 400 si OpenCV no puede leer el archivo."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap.isOpened() else 0.0
    cap.release()
    if fps <= 0 or total <= 0:
        raise HTTPException(400, "Formato no soportado: no se pudo leer el video.")
    return total / fps


def _segment_bounds(start: float, end: float) -> list[tuple[float, float]]:
    """Parte [start, end] en tramos de SEGMENT_SECONDS; el ultimo puede ser mas corto."""
    n = max(1, math.ceil(round((end - start) / SEGMENT_SECONDS, 6)))
    return [(start + i * SEGMENT_SECONDS, min(end, start + (i + 1) * SEGMENT_SECONDS))
            for i in range(n)]


def _score_segments(path: Path, bounds: list[tuple[float, float]]) -> list[float]:
    """Probabilidad de sospechoso de cada tramo, con el modelo cargado."""
    loaded = model_loader.load_model()
    scores: list[float] = []
    for i in range(0, len(bounds), INFERENCE_BATCH):
        # Cada tramo pasa por el pipeline completo: 16 frames de ESE tramo.
        clips = [preprocessor(str(path), start, end)
                 for start, end in bounds[i:i + INFERENCE_BATCH]]
        batch = torch.stack(clips).to(loaded.device)
        with torch.no_grad():
            # El modelo devuelve logits; softmax da la probabilidad de cada
            # clase y la columna 1 es "sospechoso".
            probs = loaded.model(batch).softmax(dim=1)[:, 1]
        scores.extend(probs.cpu().tolist())
    return scores


def _demo_frames(path: Path) -> list[PreprocessedFrame]:
    """Los 16 frames del video entero tras extraerlos y redimensionarlos.

    Usa los pasos 3.1 y 3.2 del pipeline, los mismos que ve el modelo. Se
    guardan como JPEG en base64 para que el frontend los muestre con <img>.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Los mismos indices que reparte extract_frames, solo para mostrar en
    # que segundo cae cada frame.
    indices = np.linspace(0, total - 1, N_FRAMES, dtype=int)
    frames_rgb = resize_and_convert(extract_frames(str(path), N_FRAMES), IMAGE_SIZE)

    result = []
    for i, (index, frame) in enumerate(zip(indices, frames_rgb)):
        # imencode espera BGR: se deshace la conversion solo para guardar.
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if ok:
            src = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
            result.append(PreprocessedFrame(index=i, time=round(index / fps, 3), src=src))
    return result


def _remember_frames(video_id: str, frames: list[PreprocessedFrame]) -> None:
    """Guarda los frames de un video y descarta el mas viejo si hay mas de DEMO_CACHE_SIZE."""
    demo_cache[video_id] = frames
    while len(demo_cache) > DEMO_CACHE_SIZE:
        demo_cache.popitem(last=False)


def _verdict(timeline: list[TimelineSegment], threshold: float) -> tuple[str, float]:
    """Veredicto global con la ventana temporal del plan.

    Es "suspicious" si los tramos sobre el umbral suman mas de
    MIN_ALERT_SECONDS seguidos, o si cubren todo lo analizado (un video de
    2-3 s no puede durar mas de 3 s). La confianza es la probabilidad del
    veredicto segun el tramo mas sospechoso: score si es sospechoso,
    1 - score si es normal.
    """
    run = longest = 0.0
    for seg in timeline:
        run = run + (seg.end - seg.start) if seg.score >= threshold else 0.0
        longest = max(longest, run)
    analyzed = timeline[-1].end - timeline[0].start
    top = max(seg.score for seg in timeline)
    if longest > MIN_ALERT_SECONDS or (longest > 0 and longest >= analyzed):
        return "suspicious", top
    return "normal", 1.0 - top


async def _analyze(request: Request, file: UploadFile | None, threshold: float | None,
                   start_sec: float | None = None,
                   end_sec: float | None = None) -> PredictResponse:
    """Valida la entrada, analiza el video (o un tramo) y arma la respuesta.

    Lo comparten /predict y /predict/clip: solo cambia el rango analizado.
    """
    content_type = request.headers.get("content-type", "")
    if content_type and not content_type.startswith("multipart/form-data"):
        raise HTTPException(415, "Se espera multipart/form-data con el video en el campo file.")
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(400, "threshold debe estar entre 0 y 1.")
    if file is None or not file.filename:
        raise HTTPException(400, "Falta el video: enviarlo en el campo file.")
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Formato no soportado: usar .mp4, .avi o .mov.")

    # El archivo temporal se borra al salir del with, pase lo que pase.
    with tempfile.TemporaryDirectory() as tmp:
        path, digest = await _save_upload(file, Path(tmp))
        duration = _video_duration(path)
        if duration > MAX_DURATION_SECONDS:
            raise HTTPException(413, "Video demasiado largo: el maximo es 2 minutos.")

        start = 0.0 if start_sec is None else start_sec
        end = duration if end_sec is None else min(end_sec, duration)
        if not 0.0 <= start < end:
            raise HTTPException(
                400, f"Rango invalido: se necesita 0 <= start_sec < end_sec <= {duration:.1f}."
            )

        bounds = _segment_bounds(start, end)
        scores = _score_segments(path, bounds)
        frames = _demo_frames(path)

    # Id corto para pedir despues los frames a /preprocessing_demo: el
    # timestamp lo hace unico aunque se suba dos veces el mismo archivo.
    video_id = f"{int(time.time())}-{digest[:8]}"
    _remember_frames(video_id, frames)
    timeline = [TimelineSegment(start=round(s, 3), end=round(e, 3), score=score)
                for (s, e), score in zip(bounds, scores)]
    verdict, confidence = _verdict(timeline, threshold)
    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        timeline=timeline,
        video_duration=round(duration, 3),
        video_id=video_id,
        threshold=threshold,
        model=model_loader.load_model().name,
        segment_seconds=SEGMENT_SECONDS,
        default_threshold=DEFAULT_THRESHOLD,
        segments=timeline,
        video=VideoInfo(name=file.filename, duration_s=round(duration, 3)),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile | None = File(None, description="Video .mp4, .avi o .mov (max 100 MB, 2 min)."),
    threshold: float | None = Form(None, description="Umbral de 0 a 1 (por defecto 0.5)."),
) -> PredictResponse:
    """Clasifica el video entero por tramos de 2 s.

    Errores: 400 si falta el video, esta vacio, no es .mp4/.avi/.mov o el
    umbral no esta en [0, 1]; 413 si pasa de 100 MB o de 2 minutos; 415 si
    el cuerpo no es multipart.
    """
    return await _analyze(request, file, threshold)


@app.post("/predict/clip", response_model=PredictResponse)
async def predict_clip(
    request: Request,
    file: UploadFile | None = File(None, description="Video .mp4, .avi o .mov (max 100 MB, 2 min)."),
    start_sec: float | None = Form(None, description="Inicio del tramo en segundos."),
    end_sec: float | None = Form(None, description="Fin del tramo en segundos."),
    threshold: float | None = Form(None, description="Umbral de 0 a 1 (por defecto 0.5)."),
) -> PredictResponse:
    """Igual que /predict pero solo sobre [start_sec, end_sec].

    El timeline usa los segundos del video original, para que el frontend lo
    dibuje en la misma barra; video_duration sigue siendo la del video entero.
    """
    # Se validan aca y no con Form(...) obligatorio: asi un cuerpo que no es
    # multipart sigue dando 415 y no un error de campo faltante.
    if start_sec is None or end_sec is None:
        content_type = request.headers.get("content-type", "")
        if not content_type or content_type.startswith("multipart/form-data"):
            raise HTTPException(400, "Faltan start_sec y end_sec.")
    return await _analyze(request, file, threshold, start_sec, end_sec)


@app.get("/preprocessing_demo", response_model=PreprocessingDemoResponse)
def preprocessing_demo(
    video_id: str = Query(..., description="El video_id que devolvio /predict."),
) -> PreprocessingDemoResponse:
    """Los 16 frames extraidos de un video ya analizado, como imagenes base64.

    Alimenta la vista del preprocesamiento paso a paso. 404 si el video_id no
    existe o ya salio del cache (se guardan los ultimos 32 videos).
    """
    frames = demo_cache.get(video_id)
    if frames is None:
        raise HTTPException(404, "video_id desconocido: analizar el video con /predict primero.")
    return PreprocessingDemoResponse(
        video_id=video_id, n_frames=len(frames), size=[IMAGE_SIZE, IMAGE_SIZE], frames=frames,
    )
