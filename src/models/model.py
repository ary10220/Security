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

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# Salida de MobileNetV2.features tras el pooling global: 1280 canales.
FEATURE_DIM = 1280
NUM_CLASSES = 2  # [Normal, Sospechoso]


class ViolenceDetector(nn.Module):
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
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.feature_extractor = mobilenet_v2(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            FEATURE_DIM, hidden_size, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def train(self, mode: bool = True) -> "ViolenceDetector":
        """Pone el modelo en modo entrenamiento sin tocar el backbone congelado.

        Con el backbone congelado hay que dejarlo siempre en eval(): sus capas
        BatchNorm actualizan estadisticas en cada forward en modo train aunque
        sus pesos no tengan gradiente, y eso lo alejaria del de ImageNet.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clasifica un lote de clips.

        Args:
            x: tensor (B, T, 3, 224, 224) normalizado con ImageNet.

        Returns:
            Logits (B, num_classes).

        Raises:
            ValueError: si x no tiene 5 dimensiones.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Se esperaba un tensor (B, T, C, H, W), llego {tuple(x.shape)}"
            )
        batch, frames = x.shape[:2]
        # Se apilan los frames de todos los clips para que el backbone los
        # procese en una sola pasada, y luego se separan de nuevo por clip.
        x = x.reshape(batch * frames, *x.shape[2:])
        feats = self.pool(self.feature_extractor(x)).reshape(batch, frames, FEATURE_DIM)

        _, (h_n, _) = self.lstm(feats)
        # h_n[-2] y h_n[-1] son el estado final de la direccion hacia adelante
        # y hacia atras. Se usan ellos y no out[:, -1]: en ese ultimo paso la
        # direccion hacia atras solo vio un frame, asi que resumiria casi nada.
        summary = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2 * hidden_size)
        return self.classifier(summary)


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


class FrameBaseline(nn.Module):
    """Baseline de 1 frame: MobileNetV2 + Linear, sin LSTM.

    Responde la pregunta "vale la pena modelar la secuencia temporal?". Si
    este modelo rinde parecido a ViolenceDetector, no hace falta la LSTM.

    TODO(Lorgio): implementar.
        - Mismo backbone MobileNetV2 congelado y el mismo pool que
          ViolenceDetector (reutilizar, no copiar con otra forma).
        - forward recibe (B, T, 3, 224, 224) como los demas modelos pero usa
          solo el frame del medio, x[:, T // 2]. Asi train.py no cambia.
        - Cabeza: Linear(FEATURE_DIM, num_classes).
        - Devuelve logits (B, num_classes).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        """Construye el baseline (pendiente)."""
        super().__init__()
        raise NotImplementedError("TODO(Lorgio): implementar FrameBaseline")


class ResNetDetector(nn.Module):
    """ResNet50 + BiLSTM: el backbone 5x mas pesado de la comparacion.

    TODO(Lorgio): implementar.
        - torchvision.models.resnet50 con ResNet50_Weights.IMAGENET1K_V2,
          quitando la capa fc (la salida de avgpool son 2048 dim por frame,
          no 1280: ajustar el input_size de la LSTM).
        - Congelar el backbone igual que ViolenceDetector, incluido el
          override de train() para no actualizar BatchNorm.
        - Misma LSTM bidireccional, mismo resumen con h_n y misma cabeza.
        - Pista: si queda casi igual que ViolenceDetector, extraer el
          cuerpo comun a una clase base en vez de duplicarlo.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        """Construye el modelo con backbone ResNet50 (pendiente)."""
        super().__init__()
        raise NotImplementedError("TODO(Lorgio): implementar ResNetDetector")


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


def save_checkpoint(model: nn.Module, name: str, path: str) -> None:
    """Guarda los pesos del modelo con lo necesario para recargarlo.

    TODO(Lorgio): implementar (serializacion, seccion 4 del plan).
        - torch.save de un dict con "model_name" (uno de MODEL_NAMES) y
          "state_dict". Guardar el nombre evita que la API tenga que
          adivinar que arquitectura corresponde al archivo.
        - La API lo carga como artifacts/model.pt; esa carpeta esta en
          .gitignore, no commitear pesos.

    Args:
        model: modelo entrenado.
        name: nombre con el que se creo (uno de MODEL_NAMES).
        path: archivo destino, por ejemplo artifacts/model.pt.
    """
    raise NotImplementedError("TODO(Lorgio): implementar save_checkpoint")


def load_checkpoint(path: str, device: str = "cpu") -> nn.Module:
    """Reconstruye un modelo desde un checkpoint de save_checkpoint.

    TODO(Lorgio): implementar.
        - Leer "model_name", crear el modelo con build_model(name,
          pretrained=False) (los pesos reales vienen del state_dict, no hace
          falta descargar ImageNet) y cargar el state_dict.
        - Devolverlo en eval() y en el device pedido. Lo usan evaluate.py y
          la API (api/main.py), asi que no debe depender de train.py.

    Args:
        path: archivo creado por save_checkpoint.
        device: "cpu" o "cuda".

    Returns:
        El modelo en modo evaluacion.
    """
    raise NotImplementedError("TODO(Lorgio): implementar load_checkpoint")
