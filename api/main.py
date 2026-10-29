"""API FastAPI: endpoints de la seccion 5 del plan.

Levantar con:
    uvicorn api.main:app --reload
y abrir http://localhost:8000/docs para probar cada endpoint.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import model_loader
from api.schemas import HealthResponse, MetricsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
