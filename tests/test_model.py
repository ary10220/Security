"""Tests de los modelos de clasificacion de video.

Se usa pretrained=False en todos los tests para no descargar los pesos de
ImageNet (~14 MB) y poder correr sin internet. La arquitectura es identica;
solo cambian los valores iniciales de los pesos.
"""

import os

import pytest
import torch

from src.models import (
    MODEL_NAMES,
    FrameBaseline,
    ResNetDetector,
    ViolenceDetector,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def _clip(batch: int = 2, frames: int = 16) -> torch.Tensor:
    """Genera un lote aleatorio con la forma que entrega el pipeline."""
    return torch.randn(batch, frames, 3, 224, 224)


def test_forward_devuelve_logits_por_clase() -> None:
    """La salida es (B, 2): un logit por clase [Normal, Sospechoso]."""
    model = ViolenceDetector(pretrained=False).eval()
    with torch.no_grad():
        out = model(_clip(batch=2))
    assert out.shape == (2, 2)
    assert out.dtype == torch.float32


def test_acepta_otra_cantidad_de_frames() -> None:
    """La LSTM no fija T: sirve para comparar 8, 16 y 32 frames."""
    model = ViolenceDetector(pretrained=False).eval()
    with torch.no_grad():
        assert model(_clip(batch=1, frames=8)).shape == (1, 2)


def test_backbone_congelado_por_defecto() -> None:
    """Con freeze_backbone=True solo entrenan la LSTM y el clasificador."""
    model = ViolenceDetector(pretrained=False)
    assert not any(p.requires_grad for p in model.feature_extractor.parameters())
    assert all(p.requires_grad for p in model.lstm.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_backbone_descongelado_entrena_todo() -> None:
    """Con freeze_backbone=False el backbone tambien recibe gradiente."""
    model = ViolenceDetector(pretrained=False, freeze_backbone=False)
    assert all(p.requires_grad for p in model.feature_extractor.parameters())


def test_backbone_congelado_no_actualiza_batchnorm() -> None:
    """En train() el backbone congelado debe seguir en eval.

    Si no, BatchNorm actualizaria sus estadisticas con cada lote aunque los
    pesos esten congelados, y el backbone dejaria de ser el de ImageNet.
    """
    model = ViolenceDetector(pretrained=False).train()
    assert not model.feature_extractor.training
    assert model.lstm.training


def test_gradiente_llega_solo_a_la_parte_entrenable() -> None:
    """Un paso de backward deja gradiente en la LSTM y no en el backbone."""
    model = ViolenceDetector(pretrained=False).train()
    loss = torch.nn.functional.cross_entropy(
        model(_clip(batch=2, frames=4)), torch.tensor([0, 1])
    )
    loss.backward()
    assert all(p.grad is not None for p in model.lstm.parameters())
    assert all(p.grad is None for p in model.feature_extractor.parameters())


def test_forward_es_determinista_en_eval() -> None:
    """En eval no hay dropout: la misma entrada da la misma salida."""
    model = ViolenceDetector(pretrained=False).eval()
    x = _clip(batch=1, frames=4)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_rechaza_entrada_con_forma_incorrecta() -> None:
    """Un clip sin dimension de lote (T, C, H, W) debe fallar con claridad."""
    model = ViolenceDetector(pretrained=False).eval()
    with pytest.raises(ValueError, match="B, T, C, H, W"):
        model(torch.randn(16, 3, 224, 224))


# ---------------------------------------------------------------------------
# Fabrica de modelos (la parte de mobilenet_lstm ya funciona)
# ---------------------------------------------------------------------------


def test_build_model_crea_el_detector_principal() -> None:
    """build_model("mobilenet_lstm") devuelve ViolenceDetector."""
    assert isinstance(build_model("mobilenet_lstm", pretrained=False), ViolenceDetector)


def test_build_model_rechaza_nombre_desconocido() -> None:
    """Un nombre fuera de MODEL_NAMES falla y lista las opciones validas."""
    with pytest.raises(ValueError, match="mobilenet_lstm"):
        build_model("vit_video")


# ---------------------------------------------------------------------------
# TODO(Lorgio): quitar cada skip cuando implementes la pieza correspondiente.
# Los cuerpos ya describen el comportamiento esperado.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="TODO(Lorgio): implementar FrameBaseline")
def test_baseline_usa_solo_el_frame_del_medio() -> None:
    """El baseline ignora los demas frames: cambiarlos no altera la salida."""
    model = FrameBaseline(pretrained=False).eval()
    x = _clip(batch=1, frames=16)
    y = x.clone()
    y[:, 0] = torch.randn_like(y[:, 0])  # cambia un frame que no es el del medio
    with torch.no_grad():
        assert model(x).shape == (1, 2)
        assert torch.equal(model(x), model(y))


@pytest.mark.skip(reason="TODO(Lorgio): implementar ResNetDetector")
def test_resnet_devuelve_logits_por_clase() -> None:
    """ResNetDetector cumple el mismo contrato que ViolenceDetector."""
    model = ResNetDetector(pretrained=False).eval()
    with torch.no_grad():
        assert model(_clip(batch=1, frames=4)).shape == (1, 2)


@pytest.mark.skip(reason="TODO(Lorgio): implementar ResNetDetector")
def test_resnet_backbone_congelado_no_actualiza_batchnorm() -> None:
    """Igual que en ViolenceDetector: train() deja el backbone en eval."""
    model = ResNetDetector(pretrained=False).train()
    assert not model.feature_extractor.training


@pytest.mark.skip(reason="TODO(Lorgio): implementar build_model para todos")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_model_crea_todos_los_nombres(name: str) -> None:
    """Cada nombre de MODEL_NAMES produce un modelo con salida (B, 2)."""
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert model(_clip(batch=1, frames=4)).shape == (1, 2)


@pytest.mark.skip(reason="TODO(Lorgio): implementar save/load_checkpoint")
def test_checkpoint_ida_y_vuelta(tmp_path) -> None:
    """Guardar y cargar devuelve un modelo con la misma salida, en eval."""
    model = build_model("mobilenet_lstm", pretrained=False).eval()
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, "mobilenet_lstm", path)
    loaded = load_checkpoint(path)
    assert not loaded.training
    x = _clip(batch=1, frames=4)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


@pytest.mark.skip(reason="TODO(Lorgio): probar con pesos reales de ImageNet")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pesos_imagenet_con_video_real(name: str) -> None:
    """Con pretrained=True y un video real, la salida es finita y (1, 2).

    Necesita internet la primera vez (descarga los pesos) y un video real en
    la variable de entorno HOCKEY_VIDEO_PATH, igual que el test opcional de
    tests/test_preprocessing.py. Al quitar el skip, que se salte solo si falta
    la variable.
    """
    from src.preprocessing import VideoPreprocessor

    video = os.environ.get("HOCKEY_VIDEO_PATH")
    if not video:
        pytest.skip("HOCKEY_VIDEO_PATH no esta definida")
    clip = VideoPreprocessor()(video).unsqueeze(0)  # (1, 16, 3, 224, 224)
    model = build_model(name, pretrained=True).eval()
    with torch.no_grad():
        out = model(clip)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()
