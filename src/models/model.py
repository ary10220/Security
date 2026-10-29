"""Modelos de clasificacion de video: normal vs sospechoso.

Arquitectura principal (seccion 4 del plan): transfer learning con
MobileNetV2 pre-entrenado en ImageNet como extractor de features por frame
y una LSTM bidireccional que mira la secuencia de 16 frames.

    Video (B, 16, 3, 224, 224)
      -> MobileNetV2 (congelado)   : vector de 1280 dim por frame
      -> BiLSTM (hidden=128)       : 128 x 2 = 256 features
      -> Dropout -> Linear(256, 64) -> ReLU -> Linear(64, 2)
      -> logits [Normal, Sospechoso]

Reparto de este archivo (fase 3 del plan, Lorgio + Diego):
    Diego  : ViolenceDetector (esta clase) y sus tests.
    Lorgio : variantes para la comparacion, fabrica de modelos y guardado
             de checkpoints. Buscar las marcas TODO(Lorgio).
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    resnet50,
)

# Salida de MobileNetV2.features tras el pooling global: 1280 canales.
FEATURE_DIM = 1280
# Salida de ResNet50 (sin avgpool ni fc) tras el pooling global: 2048 canales.
RESNET_FEATURE_DIM = 2048
NUM_CLASSES = 2  # [Normal, Sospechoso]


def _mobilenet_backbone(pretrained: bool) -> nn.Module:
    """Devuelve MobileNetV2 sin su capa de clasificacion.

    Se crea en una sola funcion para que todos los modelos con MobileNetV2
    usen exactamente el mismo backbone y la comparacion entre ellos sea justa.

    Args:
        pretrained: si carga los pesos de ImageNet.

    Returns:
        Modulo que lleva (N, 3, 224, 224) a (N, 1280, 7, 7).
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    return mobilenet_v2(weights=weights).features


def _resnet50_backbone(pretrained: bool) -> nn.Module:
    """Devuelve ResNet50 sin avgpool ni fc.

    Se quita tambien avgpool, no solo fc, para que el pooling lo haga
    _FrameBackbone.pool igual que con MobileNetV2: asi los dos backbones
    pasan por el mismo codigo y la comparacion solo cambia el backbone.

    Args:
        pretrained: si carga los pesos de ImageNet (IMAGENET1K_V2).

    Returns:
        Modulo que lleva (N, 3, 224, 224) a (N, 2048, 7, 7).
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    resnet = resnet50(weights=weights)
    # children() en orden: conv1, bn1, relu, maxpool, layer1..layer4,
    # avgpool, fc. Nos quedamos con todo menos los dos ultimos.
    return nn.Sequential(*list(resnet.children())[:-2])


class _FrameBackbone(nn.Module):
    """Base comun: un backbone de imagen pre-entrenado aplicado frame a frame.

    Junta lo que comparten todos los modelos del archivo: congelar el
    backbone, mantenerlo en eval() mientras el resto entrena, validar la forma
    de la entrada y reducir cada frame a un vector. Cada modelo concreto solo
    decide que backbone usa y que hace con esos vectores.

    Attributes:
        feature_extractor: backbone convolucional sin capa de clasificacion.
        pool: promedio global que reduce (h, w) a un vector por frame.
        freeze_backbone: si el backbone esta congelado.
    """

    def __init__(self, backbone: nn.Module, freeze_backbone: bool) -> None:
        """Registra el backbone y lo congela si se pide.

        Args:
            backbone: red convolucional que devuelve mapas (N, C, h, w).
            freeze_backbone: si es True, sus pesos no reciben gradiente.
        """
        super().__init__()
        self.feature_extractor = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "_FrameBackbone":
        """Pone el modelo en modo entrenamiento sin tocar el backbone congelado.

        Con el backbone congelado hay que dejarlo siempre en eval(): sus capas
        BatchNorm actualizan estadisticas en cada forward en modo train aunque
        sus pesos no tengan gradiente, y eso lo alejaria del de ImageNet.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Reduce cada frame a un vector de features.

        Args:
            frames: tensor (N, 3, 224, 224) con N frames sueltos.

        Returns:
            Tensor (N, C), con C los canales de salida del backbone.
        """
        return self.pool(self.feature_extractor(frames)).flatten(1)

    @staticmethod
    def _check_clip(x: torch.Tensor) -> None:
        """Falla con un mensaje claro si x no es un lote (B, T, C, H, W).

        Raises:
            ValueError: si x no tiene 5 dimensiones.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Se esperaba un tensor (B, T, C, H, W), llego {tuple(x.shape)}"
            )


class _LSTMDetector(_FrameBackbone):
    """Backbone por frame + BiLSTM + cabeza densa.

    Es el cuerpo de ViolenceDetector separado de su backbone, para que otro
    backbone lo reutilice cambiando solo la dimension de sus features.

    Attributes:
        lstm: BiLSTM sobre la secuencia de vectores por frame.
        classifier: cabeza densa que produce los logits.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        hidden_size: int,
        num_classes: int,
        freeze_backbone: bool,
    ) -> None:
        """Construye la LSTM y la cabeza sobre el backbone dado.

        Args:
            backbone: red convolucional por frame (ver _FrameBackbone).
            feature_dim: canales que devuelve el backbone, entrada de la LSTM.
            hidden_size: unidades ocultas de la LSTM (por direccion).
            num_classes: cantidad de clases de salida.
            freeze_backbone: si el backbone queda congelado.
        """
        super().__init__(backbone, freeze_backbone)
        self.lstm = nn.LSTM(
            feature_dim, hidden_size, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clasifica un lote de clips.

        Args:
            x: tensor (B, T, 3, 224, 224) normalizado con ImageNet.

        Returns:
            Logits (B, num_classes).

        Raises:
            ValueError: si x no tiene 5 dimensiones.
        """
        self._check_clip(x)
        batch, frames = x.shape[:2]
        # Se apilan los frames de todos los clips para que el backbone los
        # procese en una sola pasada, y luego se separan de nuevo por clip.
        feats = self.encode_frames(x.reshape(batch * frames, *x.shape[2:]))
        feats = feats.reshape(batch, frames, -1)

        _, (h_n, _) = self.lstm(feats)
        # h_n[-2] y h_n[-1] son el estado final de la direccion hacia adelante
        # y hacia atras. Se usan ellos y no out[:, -1]: en ese ultimo paso la
        # direccion hacia atras solo vio un frame, asi que resumiria casi nada.
        summary = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2 * hidden_size)
        return self.classifier(summary)


class ViolenceDetector(_LSTMDetector):
    """MobileNetV2 + BiLSTM para clasificar un clip de video.

    Recibe clips ya preprocesados por VideoPreprocessor y devuelve logits.
    No aplica softmax: CrossEntropyLoss lo hace internamente al entrenar, y
    la API lo aplica al predecir para obtener la probabilidad.

    Attributes:
        feature_extractor: backbone MobileNetV2 (sin la capa de clasificacion).
        pool: promedio global que reduce (7, 7) a un vector por frame.
        lstm: BiLSTM sobre la secuencia de vectores por frame.
        classifier: cabeza densa que produce los logits.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        """Construye el modelo.

        Args:
            hidden_size: unidades ocultas de la LSTM (por direccion).
            num_classes: cantidad de clases de salida.
            freeze_backbone: si es True, el backbone no se entrena. Es lo que
                se hace en la primera corrida: con pocos datos y pocos dias,
                reentrenar 2.2M de parametros de MobileNetV2 sobreajusta.
            pretrained: si es True, carga los pesos de ImageNet (descarga
                ~14 MB la primera vez). Los tests usan False para correr
                sin internet.
        """
        super().__init__(
            _mobilenet_backbone(pretrained),
            FEATURE_DIM,
            hidden_size,
            num_classes,
            freeze_backbone,
        )


# ---------------------------------------------------------------------------
# PARTE DE LORGIO (fase 3 del plan). Diego dejo la estructura y los stubs;
# cada TODO(Lorgio) dice que falta y como probarlo. Los tests que lo cubren
# estan en tests/test_model.py marcados con skip: al terminar cada pieza,
# quitar el skip y verificar que pasa.
#
# OJO, TODO(Lorgio): los tests usan pretrained=False para correr sin internet,
# pero lo que importa es que TODO funcione con los pesos REALES de ImageNet
# (pretrained=True, descarga ~14 MB de MobileNetV2 y ~100 MB de ResNet50 la
# primera vez). Nadie lo ha probado asi todavia. Antes de dar tu parte por
# cerrada, corre cada modelo con pretrained=True sobre un clip real pasado por
# VideoPreprocessor (ver test_pesos_imagenet_con_video_real) y confirma que
# da logits (1, 2) sin NaN y que el checkpoint se guarda y recarga igual.
# ---------------------------------------------------------------------------

# Nombres validos para train.py --model (seccion 4 del plan). Ariany y
# Grageda eligen el modelo con este argumento, asi que no cambiar los nombres.
MODEL_NAMES = ("mobilenet_lstm", "resnet_lstm", "mobilenet_baseline")

# TODO(Lorgio): confirmar o ajustar estos valores (tabla "Hiperparametros de
# entrenamiento" del plan). Se dejan los del plan para que train.py pueda
# arrancar; si cambias alguno, anota el por que en CONTRIBUCIONES.md.
DEFAULT_HYPERPARAMS = {
    "optimizer": "Adam",
    "lr": 1e-4,  # bajo porque la LSTM es la que aprende; backbone congelado
    "batch_size": 8,  # lo que entra en memoria con 16 x 224 x 224 x 3
    "epochs": 15,  # el plan dice 15-20; mas satura y sobreajusta
    "scheduler": "ReduceLROnPlateau",
    "early_stopping_patience": 5,
}


class FrameBaseline(_FrameBackbone):
    """Baseline de 1 frame: MobileNetV2 + Linear, sin LSTM.

    Responde la pregunta "vale la pena modelar la secuencia temporal?". Si
    este modelo rinde parecido a ViolenceDetector, no hace falta la LSTM.

    Usa el mismo backbone congelado y el mismo pool que ViolenceDetector
    (heredados de _FrameBackbone), asi la unica diferencia entre ambos es
    la parte temporal.

    Attributes:
        classifier: capa lineal de 1280 features a los logits.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        """Construye el baseline.

        Args:
            num_classes: cantidad de clases de salida.
            pretrained: si carga los pesos de ImageNet en el backbone.
        """
        # Siempre congelado: es la misma configuracion que ViolenceDetector
        # por defecto, que es contra quien se compara.
        super().__init__(_mobilenet_backbone(pretrained), freeze_backbone=True)
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clasifica cada clip mirando solo su frame del medio.

        Recibe la misma entrada que los demas modelos para que train.py no
        tenga que distinguirlos.

        Args:
            x: tensor (B, T, 3, 224, 224) normalizado con ImageNet.

        Returns:
            Logits (B, num_classes).

        Raises:
            ValueError: si x no tiene 5 dimensiones.
        """
        self._check_clip(x)
        # El frame del medio es el que tiene mas margen hacia ambos lados:
        # si la accion no empieza justo al inicio ni termina justo al final
        # del clip, es el que tiene mas chance de mostrarla.
        middle = x[:, x.shape[1] // 2]  # (B, 3, 224, 224)
        return self.classifier(self.encode_frames(middle))


class ResNetDetector(_LSTMDetector):
    """ResNet50 + BiLSTM: el backbone 5x mas pesado de la comparacion.

    Hereda de _LSTMDetector la misma LSTM, el mismo resumen con h_n, la misma
    cabeza y el mismo congelado (incluido train() con BatchNorm en eval)
    que ViolenceDetector. Lo unico que cambia es el backbone y, por eso, la
    entrada de la LSTM: 2048 features por frame en vez de 1280.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        """Construye el modelo con backbone ResNet50.

        Args:
            hidden_size: unidades ocultas de la LSTM (por direccion).
            num_classes: cantidad de clases de salida.
            freeze_backbone: si es True, el backbone no se entrena.
            pretrained: si es True, carga los pesos de ImageNet (descarga
                ~100 MB la primera vez).
        """
        super().__init__(
            _resnet50_backbone(pretrained),
            RESNET_FEATURE_DIM,
            hidden_size,
            num_classes,
            freeze_backbone,
        )


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Crea un modelo por nombre, para que train.py use --model <nombre>.

    Args:
        name: uno de MODEL_NAMES.
        pretrained: si carga los pesos de ImageNet.

    Returns:
        El modelo listo para entrenar.

    Raises:
        ValueError: si el nombre no esta en MODEL_NAMES.
    """
    if name == "mobilenet_lstm":
        return ViolenceDetector(pretrained=pretrained)
    if name == "resnet_lstm":
        return ResNetDetector(pretrained=pretrained)
    if name == "mobilenet_baseline":
        return FrameBaseline(pretrained=pretrained)
    raise ValueError(f"Modelo desconocido: {name!r}. Opciones: {MODEL_NAMES}")


def save_checkpoint(model: nn.Module, name: str, path: str | Path) -> None:
    """Guarda los pesos del modelo con lo necesario para recargarlo.

    El archivo es un dict {"model_name", "state_dict"}. Guardar el nombre
    evita que la API tenga que adivinar que arquitectura corresponde al
    archivo. La API lo carga como artifacts/model.pt; esa carpeta esta en
    .gitignore, no commitear pesos.

    Args:
        model: modelo entrenado.
        name: nombre con el que se creo (uno de MODEL_NAMES).
        path: archivo destino, por ejemplo artifacts/model.pt.

    Raises:
        ValueError: si name no esta en MODEL_NAMES.
    """
    # Se valida al guardar y no solo al cargar: con un nombre invalido el
    # error apareceria recien en la API, lejos del entrenamiento que lo causo.
    if name not in MODEL_NAMES:
        raise ValueError(f"Modelo desconocido: {name!r}. Opciones: {MODEL_NAMES}")
    torch.save({"model_name": name, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> nn.Module:
    """Reconstruye un modelo desde un checkpoint de save_checkpoint.

    No depende de train.py: lo usan la evaluacion y la API.

    Args:
        path: archivo creado por save_checkpoint.
        device: "cpu" o "cuda".

    Returns:
        El modelo en modo evaluacion, en el device pedido.
    """
    # map_location="cpu": un checkpoint entrenado en GPU tiene que poder
    # abrirse en una maquina sin GPU (la de la API). Se mueve al final.
    # weights_only=True: torch.load usa pickle, que puede ejecutar codigo al
    # cargar; asi solo se aceptan tensores y tipos basicos como este dict.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # pretrained=False: los pesos entrenados vienen del state_dict, descargar
    # los de ImageNet seria tiempo perdido porque se sobrescriben enseguida.
    model = build_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
