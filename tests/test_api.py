"""Tests de los endpoints de la API (api/main.py).

Corren sin modelo entrenado: cada test apunta ARTIFACTS_DIR a una carpeta
temporal, asi model_loader cae a pesos random y nunca lee artifacts/ real.
Los videos son los fixtures sinteticos de tests/fixtures/make_fixture.py.
"""

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api import main, model_loader
from src.models.model import build_model, save_checkpoint

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO = FIXTURES_DIR / "sample_video.mp4"  # 3 s a 30 fps
CORRUPT_VIDEO = FIXTURES_DIR / "video_corrupto.mp4"


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Carpeta de artifacts vacia y cache del modelo limpio en cada test."""
    monkeypatch.setattr(model_loader, "ARTIFACTS_DIR", tmp_path)
    model_loader.load_model.cache_clear()
    main.demo_cache.clear()
    yield tmp_path
    model_loader.load_model.cache_clear()


@pytest.fixture
def client(artifacts: Path) -> TestClient:
    """Cliente de la API; el with dispara la carga del modelo al arrancar."""
    with TestClient(main.app) as test_client:
        yield test_client


def _video(path: Path = SAMPLE_VIDEO, name: str = "clip.mp4") -> dict:
    """Arma el campo multipart `file` como lo manda el frontend."""
    return {"file": (name, path.read_bytes(), "video/mp4")}


# -----------------------------------------------------------------------------
# /health y /metrics
# -----------------------------------------------------------------------------

def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["device"] in ("cpu", "cuda")


def test_health_reports_model_loaded_status(artifacts: Path) -> None:
    """Sin checkpoint model_loaded es False; con checkpoint, True."""
    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is False

    save_checkpoint(build_model("mobilenet_lstm", pretrained=False),
                    "mobilenet_lstm", artifacts / "model_mobilenet_lstm.pt")
    model_loader.load_model.cache_clear()
    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is True


def test_checkpoint_viejo_no_tira_la_api(artifacts: Path) -> None:
    """Un .pt sin el formato de save_checkpoint cae a pesos random."""
    torch.save({"otra": "cosa"}, artifacts / "model_mobilenet_lstm.pt")
    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is False


def test_metrics_returns_json(artifacts: Path, client: TestClient) -> None:
    report = {"model": "mobilenet_lstm", "final": {"accuracy": 0.8, "f1_macro": 0.79},
              "pr_curve": None}
    (artifacts / "metrics_mobilenet_lstm.json").write_text(json.dumps(report))

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {**report, "model_size_mb": None}


def test_metrics_selects_requested_model(artifacts: Path, client: TestClient) -> None:
    """El selector lee la evaluacion indicada sin cambiar el modelo de inferencia."""
    report = {"model": "mobilenet_baseline", "final": {"accuracy": 0.66}}
    (artifacts / "metrics_mobilenet_baseline.json").write_text(json.dumps(report))
    response = client.get("/metrics?model=mobilenet_baseline")
    assert response.status_code == 200
    assert response.json()["model"] == "mobilenet_baseline"
    assert client.get("/metrics?model=desconocido").status_code == 400


def test_metrics_sin_archivo_da_404(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 404


# -----------------------------------------------------------------------------
# /predict: validaciones
# -----------------------------------------------------------------------------

def test_predict_rejects_empty_body(client: TestClient) -> None:
    assert client.post("/predict").status_code == 400


def test_predict_rejects_empty_file(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("vacio.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_predict_rejects_non_video_file(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("notas.txt", b"hola", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_unreadable_video(client: TestClient) -> None:
    """Extension valida pero contenido que OpenCV no puede abrir."""
    response = client.post("/predict", files=_video(CORRUPT_VIDEO))
    assert response.status_code == 400


def test_predict_rejects_non_multipart(client: TestClient) -> None:
    assert client.post("/predict", json={"video": "x"}).status_code == 415


@pytest.mark.parametrize("threshold", ["-0.1", "1.5", "abc"])
def test_predict_rejects_threshold_out_of_range(client: TestClient, threshold: str) -> None:
    response = client.post("/predict", files=_video(), data={"threshold": threshold})
    assert response.status_code == 400


def test_predict_rejects_video_too_big(client: TestClient,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    # Se baja el limite en vez de subir 100 MB en un test.
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    assert client.post("/predict", files=_video()).status_code == 413


def test_predict_rejects_video_too_long(client: TestClient,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "MAX_DURATION_SECONDS", 1.0)
    assert client.post("/predict", files=_video()).status_code == 413


# -----------------------------------------------------------------------------
# /predict: respuesta
# -----------------------------------------------------------------------------

def test_predict_with_valid_video_returns_prediction(client: TestClient) -> None:
    response = client.post("/predict", files=_video())
    assert response.status_code == 200
    body = response.json()

    assert body["verdict"] in ("suspicious", "normal")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["video_duration"] == pytest.approx(3.0)
    # 3 s en tramos de 2 s: [0, 2] y [2, 3].
    assert [(s["start"], s["end"]) for s in body["timeline"]] == [(0.0, 2.0), (2.0, 3.0)]
    assert all(0.0 <= s["score"] <= 1.0 for s in body["timeline"])
    # Campos del contrato del frontend.
    assert body["segments"] == body["timeline"]
    assert body["video"] == {"name": "clip.mp4", "duration_s": pytest.approx(3.0)}
    assert body["segment_seconds"] == 2.0


def test_predict_batch_returns_result_and_file_error(client: TestClient) -> None:
    """Un archivo invalido no descarta los resultados validos del lote."""
    response = client.post("/predict/batch", files=[
        ("files", ("clip.mp4", SAMPLE_VIDEO.read_bytes(), "video/mp4")),
        ("files", ("notas.txt", b"hola", "text/plain")),
    ])
    assert response.status_code == 200
    rows = response.json()
    assert rows[0]["result"]["verdict"] in ("suspicious", "normal")
    assert rows[0]["error"] is None
    assert rows[1]["result"] is None
    assert "Formato" in rows[1]["error"]


def test_verdict_aplica_la_ventana_de_3_segundos() -> None:
    """Un tramo sospechoso aislado no alcanza; dos seguidos (4 s) si."""
    def timeline(scores: list[float]) -> list[main.TimelineSegment]:
        return [main.TimelineSegment(start=2 * i, end=2 * i + 2, score=s)
                for i, s in enumerate(scores)]

    assert main._verdict(timeline([0.1, 0.9, 0.1, 0.1]), 0.5)[0] == "normal"
    assert main._verdict(timeline([0.1, 0.9, 0.8, 0.1]), 0.5) == ("suspicious", 0.9)
    # Video de un solo tramo: si es sospechoso, cubre todo lo analizado.
    assert main._verdict(timeline([0.7]), 0.5)[0] == "suspicious"


# -----------------------------------------------------------------------------
# /predict/clip y /preprocessing_demo
# -----------------------------------------------------------------------------

def test_predict_clip_analiza_solo_el_rango(client: TestClient) -> None:
    response = client.post("/predict/clip", files=_video(),
                           data={"start_sec": "1", "end_sec": "3"})
    assert response.status_code == 200
    body = response.json()
    assert [(s["start"], s["end"]) for s in body["timeline"]] == [(1.0, 3.0)]
    assert body["video_duration"] == pytest.approx(3.0)


@pytest.mark.parametrize("data", [{}, {"start_sec": "2", "end_sec": "1"},
                                  {"start_sec": "5", "end_sec": "6"}])
def test_predict_clip_rechaza_rango_invalido(client: TestClient, data: dict) -> None:
    response = client.post("/predict/clip", files=_video(), data=data)
    assert response.status_code == 400


def test_preprocessing_demo_devuelve_los_16_frames(client: TestClient) -> None:
    video_id = client.post("/predict", files=_video()).json()["video_id"]

    response = client.get("/preprocessing_demo", params={"video_id": video_id})
    assert response.status_code == 200
    body = response.json()
    assert body["n_frames"] == 16
    assert body["size"] == [224, 224]
    assert len(body["frames"]) == 16
    assert body["frames"][0]["src"].startswith("data:image/jpeg;base64,")


def test_preprocessing_demo_id_desconocido_da_404(client: TestClient) -> None:
    response = client.get("/preprocessing_demo", params={"video_id": "no-existe"})
    assert response.status_code == 404
