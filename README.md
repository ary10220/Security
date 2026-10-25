# Detector de Comportamiento Sospechoso en Cámaras de Seguridad

Proyecto académico (Tópicos y Programación Avanzada, UAGRM, 8vo semestre):
sistema que analiza fragmentos de video de cámaras de seguridad y clasifica
si contienen comportamiento sospechoso (peleas, forcejeos, asaltos) o
actividad normal.

## Stack

- **Modelo:** MobileNetV2 pre-entrenado (ImageNet) como extractor de
  features + BiLSTM para la componente temporal. Transfer learning.
- **Entrenamiento:** PyTorch 2.3, torchvision.
- **Preprocesamiento:** OpenCV + NumPy. Se extraen 16 frames espaciados
  uniformemente por clip.
- **Backend:** FastAPI + Uvicorn.
- **Frontend:** Vue 3 + Vite.

## Estructura del repo

```
.
├── data/              # Datasets (contenido ignorado por git, ver data/README.md)
├── src/
│   ├── preprocessing/ # Extraccion y normalizacion de frames
│   └── models/        # Definicion del modelo y entrenamiento
├── api/               # Servicio FastAPI
├── frontend/          # Cliente Vue
├── notebooks/         # Exploracion y prototipos
├── tests/             # pytest
├── artifacts/         # Pesos entrenados (ignorados por git)
└── docs/              # Documentacion adicional
```

## Instalación

> _Pendiente de completar en la fase 2._

Pasos aproximados:

```bash
# 1. Clonar
git clone <url-del-repo>
cd security

# 2. Entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Dependencias
pip install -r requirements.txt
```

## Equipo

Ver [`CONTRIBUCIONES.md`](CONTRIBUCIONES.md) para el detalle del aporte
de cada integrante.

- P1 — Ariany Claure
- P2 — Lorgio
- P3 — Diego
- P4 — Grageda

## Convenciones

- Reglas de código, Git y commits en [`CLAUDE.md`](CLAUDE.md).
- Commits en formato Conventional Commits (en español, título ≤60 chars).
- Nunca pushear directo a `main`: siempre PR desde una rama `feat/*` o
  `fix/*`.
