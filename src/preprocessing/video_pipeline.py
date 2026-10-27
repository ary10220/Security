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
import torch.nn.functional as F

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
    # Interpolacion AREA al achicar (lo tipico en videos de vigilancia) da
    # menos aliasing que BILINEAR; para agrandar OpenCV igual se maneja bien
    # porque los videos del dataset ya vienen a 240p o mas.
    resized = np.empty((frames.shape[0], size, size, 3), dtype=frames.dtype)
    for i, frame in enumerate(frames):
        # cv2.resize espera (ancho, alto) en ese orden, no (alto, ancho).
        bgr_resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        # El modelo se pre-entreno con imagenes RGB; OpenCV las entrega BGR
        # por herencia historica. Sin esta conversion, los canales rojo y
        # azul quedan invertidos y la red predice basura.
        resized[i] = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    return resized


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


def augment_frames(frames: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Aplica data augmentation igual a los 16 frames del clip (paso 3.4).

    Solo se usa al entrenar. El mismo flip/crop debe aplicarse a todos los
    frames del clip, o se rompe la coherencia temporal.

    Transformaciones, cada una sorteada una sola vez por clip:
        - flip horizontal con probabilidad 0.5;
        - recorte aleatorio del 80-100 % del area, reescalado al tamano original;
        - color jitter leve: brillo y contraste en [0.8, 1.2].

    Args:
        frames: tensor (T, C, H, W) ya normalizado.
        generator: generador aleatorio de torch. Con una semilla fija el
            resultado es reproducible; si es None se usa el global.

    Returns:
        Tensor (T, C, H, W) aumentado. La entrada no se modifica.
    """
    _, _, height, width = frames.shape

    def uniform(low: float, high: float) -> float:
        # Un solo sorteo por clip: todos los frames reciben el mismo valor.
        return low + (high - low) * torch.rand(1, generator=generator).item()

    # Flip horizontal: una pelea es igual de violenta vista en espejo, asi
    # que duplica las posiciones de los participantes sin cambiar la etiqueta.
    if uniform(0.0, 1.0) < 0.5:
        frames = frames.flip(-1)

    # Recorte aleatorio + resize: simula camaras con distinto encuadre o zoom.
    # Se conserva al menos el 80 % del area para no recortar la accion, y la
    # misma proporcion alto/ancho para no deformar a las personas.
    side_scale = uniform(0.8, 1.0) ** 0.5
    crop_h = max(1, round(height * side_scale))
    crop_w = max(1, round(width * side_scale))
    top = int(torch.randint(0, height - crop_h + 1, (1,), generator=generator))
    left = int(torch.randint(0, width - crop_w + 1, (1,), generator=generator))
    frames = frames[:, :, top:top + crop_h, left:left + crop_w]
    frames = F.interpolate(frames, size=(height, width), mode="bilinear",
                           align_corners=False)

    # El jitter se aplica en el espacio de pixeles [0, 1], no sobre el tensor
    # normalizado: asi "brillo x1.2" significa lo mismo en los tres canales y
    # el clamp evita valores imposibles. Despues se vuelve a normalizar.
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = frames * std + mean

    # Jitter leve (+-20 %): imita cambios de iluminacion entre camaras sin
    # borrar detalles. El contraste usa la media de todo el clip para que
    # todos los frames se ajusten igual.
    pixels = pixels * uniform(0.8, 1.2)
    clip_mean = pixels.mean()
    pixels = (pixels - clip_mean) * uniform(0.8, 1.2) + clip_mean
    pixels = pixels.clamp(0.0, 1.0)

    return ((pixels - mean) / std).float()


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convierte el arreglo NumPy a tensor PyTorch (paso 3.5).

    Args:
        frames: arreglo (T, H, W, C) normalizado.

    Returns:
        Tensor float32 de forma (T, C, H, W).
    """
    # OpenCV/NumPy guardan (T, H, W, C), pero PyTorch espera los canales
    # antes del alto y ancho: (T, C, H, W). permute reordena los ejes sin
    # copiar datos; .float() asegura float32, el tipo que usan los pesos.
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float()


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
        # Orden fijo del plan: extraer -> redimensionar/RGB -> normalizar
        # -> tensorizar -> (augmentar si entrena). Se mantiene identico en
        # train y en inferencia; la API importa esta clase tal cual y nunca
        # replica estos pasos (regla del CLAUDE.md).
        frames_bgr = extract_frames(video_path, n_frames=self.n_frames)
        frames_rgb = resize_and_convert(frames_bgr, size=self.size)
        frames_norm = normalize_frames(frames_rgb)
        tensor = to_tensor(frames_norm)

        # La augmentation se aplica DESPUES de normalizar y tensorizar
        # porque opera sobre (T, C, H, W) float; ademas asi flip/crop no
        # alteran la media/std fijadas de ImageNet. Solo en entrenamiento.
        if self.train:
            tensor = augment_frames(tensor)
        return tensor
