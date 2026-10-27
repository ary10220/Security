"""Tests del pipeline de preprocesamiento.

Se cubre cada paso por separado (extract, resize+convert, normalize,
VideoPreprocessor completo) con videos sinteticos generados por
tests/fixtures/make_fixture.py, mas un test de integracion opcional que
usa un video real de Hockey Fight si se setea la variable de entorno
HOCKEY_VIDEO_PATH.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VideoPreprocessor,
    augment_frames,
    extract_frames,
    normalize_frames,
    resize_and_convert,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VIDEO = FIXTURES_DIR / "sample_video.mp4"
SHORT_VIDEO = FIXTURES_DIR / "short_video.mp4"
CORRUPT_VIDEO = FIXTURES_DIR / "video_corrupto.mp4"

# Dimensiones del fixture sintetico. Hardcodeadas acorde con make_fixture.py:
# si alguien cambia alla el tamano, los asserts de abajo lo marcan.
FIXTURE_HEIGHT = 240
FIXTURE_WIDTH = 320


def _require_fixture(path: Path) -> None:
    """Falla el test con mensaje claro si el fixture no esta generado."""
    if not path.exists():
        pytest.fail(
            f"Falta el fixture {path}. Correr: "
            f"python tests/fixtures/make_fixture.py"
        )


# -----------------------------------------------------------------------------
# extract_frames
# -----------------------------------------------------------------------------

def test_extract_frames_returns_correct_shape() -> None:
    """Devuelve exactamente n_frames con la resolucion original del video."""
    _require_fixture(SAMPLE_VIDEO)
    frames = extract_frames(str(SAMPLE_VIDEO), n_frames=16)
    # extract_frames no redimensiona, solo muestrea: la resolucion debe ser
    # la misma que la del video original.
    assert frames.shape == (16, FIXTURE_HEIGHT, FIXTURE_WIDTH, 3)
    assert frames.dtype == np.uint8


def test_extract_frames_raises_on_missing_file(tmp_path: Path) -> None:
    """Si el archivo no existe, cv2.VideoCapture.isOpened() es False."""
    missing = tmp_path / "no_existe.mp4"
    with pytest.raises(ValueError):
        extract_frames(str(missing), n_frames=16)


def test_extract_frames_raises_on_corrupt_video() -> None:
    """Bytes aleatorios con extension .mp4 no son decodificables."""
    _require_fixture(CORRUPT_VIDEO)
    # El codigo lanza ValueError en uno de tres puntos: isOpened False,
    # total <= 0, o no se pudo leer ningun frame. Para este fixture siempre
    # cae en el primero. El test no distingue cual: solo que falla claro.
    with pytest.raises(ValueError):
        extract_frames(str(CORRUPT_VIDEO), n_frames=16)


def test_extract_frames_handles_video_shorter_than_n_frames() -> None:
    """Un video con menos frames que n_frames igual devuelve n_frames.

    Diego implemento esto repitiendo el ultimo frame leido. El contrato de
    salida (secuencia de largo fijo) es lo que el LSTM espera; sin esto,
    el batch no se puede armar.
    """
    _require_fixture(SHORT_VIDEO)
    frames = extract_frames(str(SHORT_VIDEO), n_frames=16)
    assert frames.shape[0] == 16


# -----------------------------------------------------------------------------
# resize_and_convert
# -----------------------------------------------------------------------------

def test_resize_and_convert_shape() -> None:
    """Redimensiona a (size, size) manteniendo T y canales."""
    _require_fixture(SAMPLE_VIDEO)
    frames = extract_frames(str(SAMPLE_VIDEO), n_frames=16)
    rgb = resize_and_convert(frames, size=224)
    assert rgb.shape == (16, 224, 224, 3)
    # Debe preservar el dtype para que normalize_frames despues pueda hacer
    # la division por 255 arrancando desde uint8.
    assert rgb.dtype == np.uint8


def test_bgr_to_rgb_actually_converts() -> None:
    """Un pixel BGR (255, 0, 0) -azul puro- debe quedar RGB (0, 0, 255).

    Es el test que detecta si alguien, por error, saca el cvtColor del
    pipeline. Sin esta conversion el modelo pre-entrenado da predicciones
    aleatorias porque mira los canales invertidos.
    """
    # Arreglo (T=1, H=10, W=10, 3) todo azul en BGR.
    frames_bgr = np.zeros((1, 10, 10, 3), dtype=np.uint8)
    frames_bgr[:, :, :, 0] = 255  # canal B = 255

    rgb = resize_and_convert(frames_bgr, size=10)

    # Despues del cvtColor, el azul debe estar en el canal 2 (RGB).
    assert np.all(rgb[0, :, :, 0] == 0), "canal R deberia ser 0"
    assert np.all(rgb[0, :, :, 1] == 0), "canal G deberia ser 0"
    assert np.all(rgb[0, :, :, 2] == 255), "canal B deberia ser 255 en RGB"


# -----------------------------------------------------------------------------
# normalize_frames
# -----------------------------------------------------------------------------

def test_normalize_output_range() -> None:
    """Los valores normalizados caen aproximadamente en [-2.2, 2.7].

    Cota teorica: para RGB en [0, 255] con mean/std de ImageNet,
      min = (0 - max(mean)) / min(std) = (0 - 0.485) / 0.224 ~ -2.17
      max = (1 - min(mean)) / min(std) = (1 - 0.406) / 0.224 ~ 2.65
    Dejamos un pequeno margen por seguridad.
    """
    # Frames que cubren el rango completo [0, 255] en cada canal.
    frames = np.stack(
        [
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.full((10, 10, 3), 255, dtype=np.uint8),
        ]
    )
    normalized = normalize_frames(frames)

    assert normalized.dtype == np.float32
    assert normalized.min() >= -2.5
    assert normalized.max() <= 2.7


def test_normalize_uses_imagenet_stats() -> None:
    """Verifica el valor exacto para un frame gris medio (127/255 por canal).

    Con un pixel de valor 127 en los 3 canales, despues de /255 queda ~0.498.
    La normalizacion resta mean y divide std, canal por canal: los tres
    valores esperados son distintos porque mean y std difieren por canal.
    """
    frames = np.full((1, 4, 4, 3), 127, dtype=np.uint8)
    normalized = normalize_frames(frames)

    scaled = 127.0 / 255.0
    expected = (
        (scaled - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (scaled - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (scaled - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    )
    # Todos los pixeles del canal c deben estar en expected[c].
    for c in range(3):
        assert np.allclose(normalized[0, :, :, c], expected[c], atol=1e-4), (
            f"Canal {c}: esperado {expected[c]:.4f}, "
            f"obtenido {normalized[0, 0, 0, c]:.4f}"
        )


# -----------------------------------------------------------------------------
# augment_frames
# -----------------------------------------------------------------------------

def _random_clip(seed: int = 0) -> torch.Tensor:
    """Clip (T=16, C=3, H=32, W=32) ya normalizado, como sale de to_tensor."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(16, 32, 32, 3), dtype=np.uint8)
    normalized = normalize_frames(frames)
    return torch.from_numpy(normalized).permute(0, 3, 1, 2).float()


def test_augment_preserves_shape_and_dtype() -> None:
    """El modelo espera (T, C, H, W) float32: la augmentation no lo cambia."""
    clip = _random_clip()
    out = augment_frames(clip, generator=torch.Generator().manual_seed(0))
    assert out.shape == clip.shape
    assert out.dtype == torch.float32


def test_augment_applies_same_transform_to_every_frame() -> None:
    """Si los 16 frames son iguales, despues de aumentar siguen iguales.

    Es el test de coherencia temporal: si flip, crop o jitter se sortearan
    por frame, la secuencia tendria saltos que el LSTM leeria como movimiento.
    """
    frame = _random_clip()[0:1]
    clip = frame.repeat(16, 1, 1, 1)
    for seed in range(10):
        out = augment_frames(clip, generator=torch.Generator().manual_seed(seed))
        assert torch.allclose(out, out[0:1].expand_as(out)), f"semilla {seed}"


def test_augment_changes_the_clip() -> None:
    """Con una semilla dada el clip sale modificado (no es la identidad)."""
    clip = _random_clip()
    out = augment_frames(clip, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(out, clip)


def test_augment_is_reproducible_with_same_seed() -> None:
    """Misma semilla, mismo resultado: necesario para repetir experimentos."""
    clip = _random_clip()
    out_a = augment_frames(clip, generator=torch.Generator().manual_seed(7))
    out_b = augment_frames(clip, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out_a, out_b)


def test_augment_does_not_modify_input() -> None:
    """La entrada no se modifica in-place (el llamador puede reutilizarla)."""
    clip = _random_clip()
    original = clip.clone()
    augment_frames(clip, generator=torch.Generator().manual_seed(0))
    assert torch.equal(clip, original)


def test_augment_keeps_imagenet_range() -> None:
    """El jitter no saca los valores del rango posible tras normalizar."""
    clip = _random_clip()
    for seed in range(10):
        out = augment_frames(clip, generator=torch.Generator().manual_seed(seed))
        assert out.min().item() >= -2.5
        assert out.max().item() <= 2.7


# -----------------------------------------------------------------------------
# VideoPreprocessor (integracion)
# -----------------------------------------------------------------------------

def test_video_preprocessor_integration_with_real_video() -> None:
    """El pipeline completo sobre el fixture sintetico produce el tensor esperado."""
    _require_fixture(SAMPLE_VIDEO)
    preprocessor = VideoPreprocessor(n_frames=16, size=224, train=False)
    tensor = preprocessor(str(SAMPLE_VIDEO))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (16, 3, 224, 224)
    assert tensor.dtype == torch.float32
    # Mismo rango teorico que normalize_frames.
    assert tensor.min().item() >= -2.5
    assert tensor.max().item() <= 2.7


def test_video_preprocessor_train_mode_applies_augmentation() -> None:
    """Con train=True el pipeline aumenta el clip; con train=False no.

    Se compara contra la salida de inferencia del mismo video: si alguien
    desconecta augment_frames del pipeline, ambas salidas serian iguales.
    """
    _require_fixture(SAMPLE_VIDEO)
    plain = VideoPreprocessor(n_frames=16, size=224, train=False)(str(SAMPLE_VIDEO))

    torch.manual_seed(0)
    tensor = VideoPreprocessor(n_frames=16, size=224, train=True)(str(SAMPLE_VIDEO))

    assert tensor.shape == (16, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= -2.5
    assert tensor.max().item() <= 2.7
    assert not torch.allclose(tensor, plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere GPU con CUDA.")
def test_augment_works_on_gpu() -> None:
    """Un clip en GPU se aumenta sin errores y el resultado queda en GPU."""
    clip = _random_clip().to("cuda")
    out = augment_frames(clip, generator=torch.Generator(device="cuda").manual_seed(0))
    assert out.device.type == "cuda"


HOCKEY_VIDEO_PATH = os.environ.get("HOCKEY_VIDEO_PATH")


@pytest.mark.skipif(
    not HOCKEY_VIDEO_PATH or not Path(HOCKEY_VIDEO_PATH).exists(),
    reason="Setear HOCKEY_VIDEO_PATH apuntando a un .avi de Hockey Fight.",
)
def test_video_preprocessor_integration_with_hockey_fight() -> None:
    """Validacion end-to-end sobre un video real del dataset Hockey Fight."""
    preprocessor = VideoPreprocessor(n_frames=16, size=224, train=False)
    tensor = preprocessor(HOCKEY_VIDEO_PATH)

    assert tensor.shape == (16, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= -2.5
    assert tensor.max().item() <= 2.7
