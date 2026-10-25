# Datos

Esta carpeta contiene los datasets de video del proyecto. **El contenido de
`raw/` y `processed/` NO se sube al repositorio** (está en `.gitignore`):
son archivos pesados y, en el caso de datasets externos, no corresponde
redistribuirlos. Cada integrante los descarga localmente siguiendo las
instrucciones de abajo.

## Estructura

```
data/
├── raw/          # Videos originales, sin modificar. NO commitear.
└── processed/    # Frames y tensores precomputados. NO commitear.
```

### `raw/`

Videos originales tal cual se descargaron de la fuente. **No modificar
estos archivos**: el pipeline de preprocesamiento siempre arranca desde
aquí, por lo que si se corrompen o se editan a mano, los resultados dejan
de ser reproducibles.

Convención de nombres sugerida:

```
raw/
├── fights/                # Clase: pelea / forcejeo
│   ├── fight_0001.mp4
│   └── fight_0002.mp4
└── normal/                # Clase: comportamiento normal
    ├── normal_0001.mp4
    └── normal_0002.mp4
```

### `processed/`

Salida del pipeline de extracción de frames (ver `src/preprocessing/`).
Contiene los 16 frames muestreados por video, redimensionados a 224x224
(entrada que espera MobileNetV2), serializados en `.npy` o `.pt`.

Esta carpeta se regenera corriendo el pipeline, no se versiona.

## Fuentes de datos

Pendiente de definir con el equipo. Candidatos iniciales:

- **UCF Crime Dataset** — vigilancia real, con y sin eventos violentos.
- **RWF-2000** — 2000 clips cortos balanceados fight/no-fight.
- **Hockey Fight Dataset** — baseline chico para pruebas rápidas.

Cuando se decida cuál usar, agregar aquí el link oficial, el checksum del
archivo descargado y una línea con la licencia.

## Cómo descargar

(A completar cuando se elija el dataset.) La idea es un script en
`scripts/download_data.sh` o un notebook que haga la descarga + verificación
de hash, de forma que cualquier integrante pueda replicar el entorno.
