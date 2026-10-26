"""Pipeline de preprocesamiento de video.

Convierte un video en un tensor listo para el modelo (MobileNetV2 + BiLSTM).
El mismo pipeline se usa al entrenar y al predecir en la API: la API nunca
reimplementa estos pasos, solo importa esta clase.

Pasos (ver seccion 3 del plan):
    3.1 extract_frames        -> 16 frames espaciados uniformemente
    3.2 resize_and_convert    -> 224x224 y BGR -> RGB
    3.3 normalize_frames      -> escala a [0, 1] y normaliza con ImageNet
    3.4 augment_frames        -> solo en entrenamiento
    3.5 to_tensor             -> tensor (T, C, H, W)
    3.6 VideoPreprocessor     -> encadena todo
"""

import cv2
import numpy as np
import torch

# Estadisticas de ImageNet: el backbone se pre-entreno con ellas, asi que
# hay que normalizar igual o las predicciones se vuelven aleatorias.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

N_FRAMES = 16  # estandar en la literatura (I3D, SlowFast, CLIP-TSA)
IMAGE_SIZE = 224  # tamano de entrada de MobileNetV2 / ResNet


def extract_frames(video_path: str, n_frames: int = N_FRAMES) -> np.ndarray:
    """Extrae un numero fijo de frames espaciados uniformemente (paso 3.1).

    Args:
        video_path: ruta al video (.mp4, .avi o .mov).
        n_frames: cantidad de frames a extraer.

    Returns:
        Arreglo de forma (n_frames, alto, ancho, 3) en formato BGR.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"El video no tiene frames legibles: {video_path}")

    # linspace reparte los indices parejo de principio a fin del clip, asi
    # los 16 frames cubren todo el movimiento y no solo el comienzo.
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No se pudo leer ningun frame de: {video_path}")

    # Si algun frame falla al leerse, se repite el ultimo valido para
    # conservar siempre n_frames (el LSTM espera una secuencia de largo fijo).
    while len(frames) < n_frames:
        frames.append(frames[-1])

    return np.array(frames)  # forma (n_frames, alto, ancho, 3) en BGR


def resize_and_convert(frames: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Redimensiona cada frame y convierte BGR a RGB (paso 3.2).

    Args:
        frames: arreglo (T, alto, ancho, 3) en BGR, salida de extract_frames.
        size: lado del cuadrado de salida.

    Returns:
        Arreglo (T, size, size, 3) en RGB.
    """
    # TODO (Ariany): cv2.resize + cv2.cvtColor(COLOR_BGR2RGB).
    raise NotImplementedError


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Escala a [0, 1] y normaliza con media/desvio de ImageNet (paso 3.3).

    Args:
        frames: arreglo (T, size, size, 3) en RGB con valores 0-255.

    Returns:
        Arreglo float32 (T, size, size, 3) normalizado.
    """
    # Se trabaja en float32: con uint8 la resta de la media se desbordaria.
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    # Primero a rango [0, 1] y luego se centra y escala con las estadisticas
    # de ImageNet. Es obligatorio: el backbone se pre-entreno asi y, sin esto,
    # el modelo da predicciones aleatorias. Se aplica por canal (broadcasting).
    frames = frames.astype(np.float32) / 255.0
    return (frames - mean) / std


def augment_frames(frames: torch.Tensor) -> torch.Tensor:
    """Aplica data augmentation igual a los 16 frames del clip (paso 3.4).

    Solo se usa al entrenar. El mismo flip/crop debe aplicarse a todos los
    frames del clip, o se rompe la coherencia temporal.

    Args:
        frames: tensor (T, C, H, W) ya normalizado.

    Returns:
        Tensor (T, C, H, W) aumentado.
    """
    # TODO (Ariany): flip horizontal (p=0.5), random crop + resize, color jitter leve.
    raise NotImplementedError


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convierte el arreglo NumPy a tensor PyTorch (paso 3.5).

    Args:
        frames: arreglo (T, H, W, C) normalizado.

    Returns:
        Tensor float32 de forma (T, C, H, W).
    """
    # TODO (Diego): torch.from_numpy(...).permute(0, 3, 1, 2).float()
    raise NotImplementedError


class VideoPreprocessor:
    """Encadena todos los pasos del pipeline (paso 3.6).

    Attributes:
        n_frames: frames a extraer por video.
        size: lado de la imagen de salida.
        train: si es True, aplica data augmentation.
    """

    def __init__(self, n_frames: int = N_FRAMES, size: int = IMAGE_SIZE,
                 train: bool = False) -> None:
        self.n_frames = n_frames
        self.size = size
        self.train = train

    def __call__(self, video_path: str) -> torch.Tensor:
        """Procesa un video completo.

        Args:
            video_path: ruta al video.

        Returns:
            Tensor (n_frames, 3, size, size) listo para el modelo.
        """
        # TODO (ambos, al final): encadenar los pasos 3.1 a 3.5.
        raise NotImplementedError
