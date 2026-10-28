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
