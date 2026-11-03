# Contexto del proyecto

Proyecto académico: detector de comportamiento sospechoso en cámaras de
seguridad (peleas, forcejeos, asaltos). Materia Tópicos y Programación
Avanzada, UAGRM. Entrega 2 de noviembre 2026, presentación 3 de noviembre.

Grupo de 4 integrantes. El ingeniero evalúa commits individuales — cada
integrante debe poder EXPLICAR cualquier código que subió.

Plan completo en el doc "Plan Proyecto Integrador — Detección de
Comportamiento Sospechoso en Cámaras de Seguridad" compartido aparte.

## Stack

- Python 3.10+, PyTorch 2.3, torchvision, OpenCV, NumPy
- Backend: FastAPI + uvicorn
- Frontend: Vue 3 + Vite
- Modelo: MobileNetV2 pre-entrenado (ImageNet) + BiLSTM, transfer learning

## Reglas de código

- **Comentarios obligatorios en ESPAÑOL** en cada función, explicando qué
  hace y por qué. No comentar lo obvio (`i += 1  # incrementa i`), sí
  explicar decisiones ("extraemos 16 frames porque es el estándar de I3D
  y balancea costo/información").
- Docstrings en formato Google style para toda función pública.
- Nombres de variables en inglés (`frames`, `model`), textos de usuario
  y comentarios en español.
- Type hints en todas las funciones.

## Reglas de Git — CRÍTICAS

- NUNCA pushear directo a `main`. Siempre crear rama `feat/*` o `fix/*`
  y abrir Pull Request.
- Una rama por FEATURE, no por persona. Nombre corto: `feat/preprocess-frames`,
  no `feat/trabajo-de-princesa`.
- Mensajes de commit en formato Conventional Commits, CORTOS (≤60
  caracteres en el título):
  - `feat(preprocess): extraer 16 frames con opencv`
  - `fix(api): validar tamano maximo de video`
  - `docs(readme): agregar instrucciones de instalacion`
- Un commit = un cambio lógico. No commits gigantes.
- Cada 1-2 horas de trabajo = un commit.
- Cuando trabajás en pareja con otra integrante, agregar al final:
  `Co-authored-by: Nombre <email@dominio.com>`

## Qué NO hacer

- No `git push --force` en main.
- No cambiar el autor de los commits de otra persona.
- No borrar archivos de otras ramas sin preguntar.
- No commitear modelos entrenados (`.pt`, `.pth`), datasets crudos, `.venv/`,
  `__pycache__/` — todo eso va al `.gitignore`.
- No reimplementar el preprocesamiento en la API: la API carga el pipeline
  serializado y lo usa tal cual.

## Estructura esperada

Ver el plan compartido. Las carpetas principales son `data/`, `src/preprocessing/`,
`src/models/`, `api/`, `frontend/`, `tests/`, `docs/`, `notebooks/`.

## Integrantes

- P1 — Ariany
- P2 — Lorgio
- P3 — Diego
- P4 — Grageda

## Estado del proyecto (actualizado el 2 nov 2026)

Resumen para quien retome el trabajo con Claude. Es un estado, no un plan:
si algo cambia, actualizar esta sección en el mismo PR.

### Hecho
- **Datos:** RWF-2000 (2000 clips, Zenodo, CC BY 4.0) y Real Life Violence
  (2000 clips, Kaggle) están en el Drive del grupo en `datasets/`. Hockey Fight
  fue descargado por Diego. Detalle, licencias y estructura real en
  [`data/README.md`](data/README.md).
- **Preprocesamiento:** `src/preprocessing/video_pipeline.py` implementa los
  6 pasos (extraer 16 frames, resize+RGB, normalizar, augmentation solo con
  `train=True`, tensor `(16, 3, 224, 224)`, clase `VideoPreprocessor`), con
  tests en `tests/test_preprocessing.py`. `augment_frames` (Diego) sortea
  flip, recorte y jitter una vez por clip.
- **EDA (Grageda):** `src/preprocessing/dataset_metadata.py` (metadatos de
  videos y `threshold_shortcut`), `notebooks/02_eda_datasets.ipynb` y
  `notebooks/03_comparar_frames.ipynb`. La revisión y el complemento de Lorgio
  están en `CONTRIBUCIONES.md`.
- **Modelos, API y frontend:** `src/models/`, `api/` y `frontend/` implementados.
- **Entrenamiento:** dos corridas, ambas en `docs/informe.md`. (1) CPU, subset de
  una copia parcial de RWF-2000, con `scripts/train_local_cpu.py`: los tres
  modelos, métricas en `artifacts/cpu/`. La semilla elige los clips según la
  lista de archivos, así que con otra copia salen otros clips. (2) GPU,
  RWF-2000 completo, con `src/models/train.py`: los tres modelos, 400 clips de
  test, 73,0 % / 73,5 % / 74,3 % (`mobilenet_lstm` / `mobilenet_baseline` /
  `resnet_lstm`), métricas en `artifacts/`. La regla de resolución acierta 75 %
  en esos mismos clips: ningún modelo la supera, y por resolución el acierto es
  compatible con que la usen (informe §5.2). No presentar 73–74 % como prueba de
  que reconocen peleas.
- **Modos de riesgo (PR #31):** la API y `train.py --mode` soportan `peleas`,
  `interior` y `exterior`. Interior y exterior se entrenaron con 78 videos de
  ASHEN (Robbery + Assault anotados y normales); datos, decisiones y comandos
  en `data/ashen/`. Test: interior 75,8 % (62 clips de 18 videos); exterior
  (13 clips de 4 videos) no tiene cifra: es inconcluso. Los normales de ASHEN
  son todos 320x240 y más livianos que los de riesgo (sesgo de selección). Ver
  `docs/informe.md` §5.4.
- **Predicciones por clip:** `artifacts/predicciones_<modelo>_test.csv` para
  los 3 modelos de peleas (400 clips) y los 2 modos; la matriz coincide con su
  JSON de métricas.

### Pendiente
- Exterior necesita más videos fuente para que sus métricas digan algo.
- Rehacer los normales de ASHEN con resolución y duración comparables a los de
  riesgo y volver a medir si los modos aprenden el estilo de imagen en vez del
  comportamiento (atajo).
- Grageda (PR #33, #34, #36): comparación 8/16/32 frames, regla de resolución y
  auditoría de una muestra de `data/ashen/videos.csv`.
- Decidir qué modelo usa la demo de peleas: hoy carga `mobilenet_lstm`, que es el
  de menor recall de Fight (0,62) de los tres.
- Comparación de **exactitud** entre 8, 16 y 32 frames: requiere entrenar con
  cada valor; los notebooks solo miden cobertura y costo.

### Hallazgos del EDA que afectan al modelo (medidos, ver notebook 02)
- **RWF-2000:** 5,0 s a 30 fps en todos los clips, `.avi`. Resolución muy
  variable (700 clips de 320x240, 628 de 1280x720, 31 de 1920x1080).
- **Atajos:** una regla de un solo umbral sobre metadatos separa las clases
  mejor que el azar. En Real Life: fps 81,3 % y resolución ~77 %. En RWF-2000:
  resolución ~75 %, bitrate ~68 %, y fps/duración 50 % (no delatan).
  Un modelo con buena exactitud no demuestra por sí solo que aprendió el
  comportamiento: comparar contra estas líneas base y revisar errores por
  resolución.
- **Fugas entre conjuntos:** los clips de RWF-2000 salen de ~961 videos fuente
  (prefijo del nombre `<id>_<n>.avi`) y ninguna fuente está en `train` y `val`
  a la vez. **Cualquier split nuevo o validación cruzada debe dividir por
  fuente, no por clip.** Real Life no trae split ni fuente.
- **Real Life:** 3 clips duran más de 30 s (`V_789.mp4`, `NV_992.mp4`,
  `V_398.mp4`); con muestreo uniforme quedan con frames muy separados.
  Mezcla deportes de contacto y tomas de transmisión, no solo vigilancia.

### Decisiones abiertas (las toma el equipo, no Claude)
1. ¿Se excluyen o se recortan los 3 clips largos de Real Life?
2. ¿Se confirma 16 frames? Los notebooks lo respaldan en costo y cobertura,
   pero no en exactitud.
3. ¿Cómo se parte Real Life en train/val/test y cómo se reportan los
   resultados por dataset (RWF-2000 y Real Life por separado)?
4. ~~¿Quién implementa `augment_frames`?~~ Resuelto: lo hizo Diego, según el
   plan (día 4).
5. Duración del segmento de análisis (el plan menciona 2 s y 3 s).
6. Versión de Python del equipo: `requirements.txt` fija `torch==2.3.1`
   (Python 3.10–3.12).

### Cómo trabajar con los datos y notebooks
- Poner los videos en `data/raw/rwf2000/` y `data/raw/real_life/` como indica
  `data/README.md` (nunca se commitean). Ejecutar el notebook 02 antes que el
  03: el 02 genera `data/processed/metadata_*.csv` que el 03 reutiliza.
- Los notebooks se suben **ejecutados** (con salidas). Antes de commitear,
  comprobar que las salidas no incluyan rutas absolutas de la máquina local.
- No afirmar métricas que no se hayan medido: las cifras del plan son
  estimaciones, no resultados del proyecto.