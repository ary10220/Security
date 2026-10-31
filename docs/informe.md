# Informe técnico — Detección de comportamiento sospechoso en cámaras de seguridad

Proyecto integrador de Tópicos y Programación Avanzada (UAGRM).

Cada integrante redacta sus secciones en su propia rama y las enlaza con el
código y los commits que las respaldan. Las secciones marcadas como
_Pendiente_ todavía no tienen autor.

## 1. Problema

_Pendiente._ Un párrafo: inseguridad en Bolivia y por qué priorizar videos de
cámaras.

## 2. Dataset

_Pendiente._ Fuentes (RWF-2000, Real Life Violence), licencias, tamaños, split
y por qué no se usan cámaras ajenas. Ver `data/README.md`.

## 3. Preprocesamiento

El modelo no recibe videos, recibe tensores. El módulo
`src/preprocessing/video_pipeline.py` convierte un video en un tensor de forma
`(16, 3, 224, 224)` con seis pasos. La clase `VideoPreprocessor` los encadena
y es la única implementación: el entrenamiento y la API la importan tal cual, de
modo que un video recibe exactamente el mismo tratamiento al entrenar y al
predecir.

| Paso | Función | Qué hace |
|---|---|---|
| 3.1 | `extract_frames` | Extrae 16 frames espaciados uniformemente con OpenCV |
| 3.2 | `resize_and_convert` | Redimensiona a 224×224 y pasa de BGR a RGB |
| 3.3 | `normalize_frames` | Escala a [0, 1] y normaliza con media y desvío de ImageNet |
| 3.4 | `augment_frames` | Variaciones aleatorias, solo al entrenar |
| 3.5 | `to_tensor` | Reordena a `(T, C, H, W)` |
| 3.6 | `VideoPreprocessor` | Encadena los pasos anteriores |

### 3.1 Extracción de frames

Se toman 16 frames con índices equiespaciados en lugar de todos: un clip de
5 s a 30 fps tiene 150 frames muy parecidos entre sí, y procesarlos todos
multiplicaría el costo sin aportar información nueva. Se eligió 16 por ser el
valor habitual en la literatura de clasificación de video (I3D, SlowFast,
CLIP-TSA).

Los notebooks de EDA (`notebooks/03_comparar_frames.ipynb`) respaldan esa
elección en **costo y cobertura temporal**, pero no en **exactitud**: medirla
exige entrenar con 8, 16 y 32 frames, y esa comparación sigue abierta.

La función acepta `start_sec` y `end_sec`. La API la usa para analizar un
video largo por tramos de 2 s con este mismo pipeline, sin reimplementar la
extracción.

### 3.2 Redimensionado y conversión de color

Cada frame pasa a 224×224, el tamaño de entrada de MobileNetV2 y ResNet50. Se
usa interpolación `INTER_AREA`, que produce menos aliasing que la bilineal al
reducir (los clips de RWF-2000 miden 320×240 o más).

OpenCV entrega los canales en orden BGR y las redes preentrenadas esperan RGB.
Sin la conversión, el rojo y el azul quedan invertidos y la red recibe colores
que nunca vio al entrenar.

### 3.3 Normalización

Los píxeles se dividen por 255 y luego se les resta la media y se divide por el
desvío de ImageNet, por canal:

- media = (0.485, 0.456, 0.406)
- desvío = (0.229, 0.224, 0.225)

Es obligatorio porque el backbone se preentrenó con esos valores. Con otra
escala, las activaciones internas no corresponden a lo que la red aprendió y las
predicciones dejan de tener sentido. Se calcula en `float32`: con `uint8`, la
resta de la media se desbordaría.

### 3.4 Data augmentation (solo en entrenamiento)

`augment_frames` sortea **una sola vez por clip** tres transformaciones y las
aplica igual a los 16 frames:

- **Flip horizontal**, con probabilidad 0.5: una pelea espejada sigue siendo
  una pelea.
- **Recorte aleatorio** del 80 al 100 % del área, reescalado al tamaño original:
  simula una cámara con otro encuadre sin recortar la acción.
- **Jitter leve de brillo y contraste** (factores entre 0.8 y 1.2): da
  robustez ante cámaras más oscuras o más claras. Se aplica sobre los píxeles
  en [0, 1], antes de volver a normalizar, para que "brillo ×1.2" signifique lo
  mismo en los tres canales.

Se sortea una sola vez por clip porque un flip o un recorte distinto en cada
frame rompería la coherencia temporal, que es justo lo que la LSTM necesita para
ver el movimiento. No se usan rotaciones ni flip vertical: las cámaras de
seguridad están derechas y rotar inventaría escenas imposibles. La función
acepta un `torch.Generator`, así que los tests pueden reproducir el resultado.

### 3.5 y 3.6 Tensorización y encadenado

`to_tensor` reordena los ejes a `(T, C, H, W)`, la forma que espera el modelo
CNN + LSTM. `VideoPreprocessor` aplica los pasos en este orden fijo: extraer,
redimensionar y convertir a RGB, normalizar, tensorizar y, si `train=True`,
augmentar. La augmentation va al final porque opera sobre el tensor ya
normalizado; en inferencia no se aplica.

Los tests (`tests/test_preprocessing.py`) cubren la forma y el rango de la
salida, la conversión BGR→RGB, la coherencia del augmentation entre frames, su
reproducibilidad y su funcionamiento en CPU y GPU.

## 4. Modelo

El problema se resuelve con **transfer learning**: en lugar de entrenar una red
para video desde cero (lo que exige días de GPU y millones de clips), se reutiliza
una red de imágenes ya entrenada y solo se entrena la parte que entiende el
movimiento.

### 4.1 Arquitectura

La clase `ViolenceDetector` (`src/models/model.py`) tiene tres bloques:

1. **MobileNetV2 preentrenado en ImageNet**, aplicado a cada frame. Extrae un
   vector de 1280 números por frame. Está **congelado**: sus pesos no se
   modifican.
2. **BiLSTM** (una capa, 128 unidades por dirección). Recibe la secuencia de 16
   vectores y produce 256 features que resumen el clip.
3. **Cabeza de clasificación**: `Dropout(0.3)` → `Linear(256→64)` → `ReLU` →
   `Linear(64→2)`. Devuelve dos logits, uno por clase `[Normal, Sospechoso]`; la
   probabilidad se obtiene con softmax en la API.

```
Clip (16, 3, 224, 224)
  → MobileNetV2 por frame (congelado)   → 16 vectores de 1280
  → BiLSTM (128 por dirección)          → 256 features
  → Dropout → Linear(256→64) → ReLU → Linear(64→2)
  → logits [Normal, Sospechoso]
```

### 4.2 Decisiones de diseño

- **Por qué MobileNetV2:** es liviano y apunta a correr en una laptop sin GPU.
  La hipótesis es que un backbone más pesado como ResNet50 gana poca exactitud
  a costa de mucho más tiempo de inferencia; la comparación de la sección 4.3
  la confirma o la desmiente.
- **Por qué congelar el backbone:** con pocos datos y pocos días, reentrenar
  toda la red sobreajusta. Además, el backbone se mantiene en modo `eval()`
  durante el entrenamiento: sus capas BatchNorm actualizarían sus estadísticas en
  cada pasada en modo `train` aunque los pesos estén congelados, y eso lo
  alejaría de ImageNet.
- **Por qué una LSTM:** un frame suelto de una pelea puede parecer un abrazo; el
  movimiento solo se ve mirando varios frames juntos.
- **Por qué bidireccional:** el modelo entiende qué pasó antes y después de cada
  momento.
- **Qué se le pasa al clasificador:** el estado final de la LSTM en las dos
  direcciones (`h_n`), no la salida del último paso. En ese último paso, la
  dirección hacia atrás solo vio un frame, así que resumiría casi nada del clip.

### 4.3 Comparación de modelos

Se entrenan tres variantes con el mismo dataset, el mismo split y la misma
métrica, y el modelo que queda en producción es el que gane la comparación:

| Nombre (`--model`) | Qué prueba |
|---|---|
| `mobilenet_baseline` | ¿Vale la pena modelar la secuencia? Un solo frame, sin LSTM |
| `mobilenet_lstm` | Candidato principal: backbone liviano + BiLSTM |
| `resnet_lstm` | ¿Vale la pena un backbone más pesado? |

Los resultados de esta comparación se reportan en la sección 5, una vez
entrenados los tres modelos. Esta sección no incluye cifras de rendimiento
porque todavía no se midieron.

### 4.4 Hiperparámetros de entrenamiento

Valores por defecto en `DEFAULT_HYPERPARAMS` (`src/models/model.py`):

| Hiperparámetro | Valor | Motivo |
|---|---|---|
| Optimizador | Adam | Estándar, converge rápido |
| Learning rate | 1e-4 | Bajo, porque solo aprende la LSTM y la cabeza |
| Batch size | 8 | Lo que entra en memoria con 16×224×224×3 |
| Épocas | 15 | Más satura y sobreajusta |
| Scheduler | ReduceLROnPlateau | Baja el lr si la métrica de validación se estanca |
| Early stopping | paciencia 5 | Corta si no mejora |

## 5. Resultados

_Pendiente._ Depende de los modelos entrenados: métricas, matriz de confusión,
3 falsos positivos y 3 falsos negativos con explicación. No se incluyen cifras
hasta medirlas.

## 6. Arquitectura del sistema

_Pendiente._ Diagrama de los 5 endpoints y flujo de una petición.

## 7. Limitaciones y mitigaciones

_Pendiente._ Niños jugando, umbral ajustable, ventana temporal.

## 8. Uso de asistentes de IA

_Pendiente._ Declaración honesta de quién usó qué herramienta y para qué. Cada
integrante declara la suya; ver también `CONTRIBUCIONES.md`.
