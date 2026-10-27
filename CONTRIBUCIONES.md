# Contribuciones

Registro individual del trabajo de cada integrante del grupo. La idea es
que al momento de la evaluación cada una pueda señalar los commits y las
partes del código que redactó, y que el ingeniero vea con claridad el
reparto.

Reglas:

- Cada integrante llena **su propia sección**.
- Enlazar commits con el hash corto (p.ej. `a1b2c3d`) o PRs con `#NN`.
- Nombrar los archivos/módulos sobre los que se trabajó, no sólo "ayudé".
- Si se trabajó en pareja, usar `Co-authored-by:` en el commit y
  mencionarlo acá en ambas secciones.

---

## P1 — Ariany Claure

Fase de setup (días 1-2, PR #1):

- **Rama `feat/setup-estructura-inicial`:** estructura inicial del repo
  (carpetas `src/`, `api/`, `frontend/`, `tests/`, etc. con sus `__init__.py`
  y `.gitkeep`), `requirements.txt` con versiones fijas compatibles con
  Python 3.10+ (ojo con `numpy==1.26.4` para no romper torch 2.3), `.gitignore`
  ampliado (`artifacts/*.pt`, `data/raw/`, `data/processed/`, `node_modules/`),
  `data/README.md` inicial, `README.md` del proyecto, `CONTRIBUCIONES.md` y
  `CLAUDE.md` con las reglas.
- **Notebook `notebooks/01_explore_dataset.ipynb`:** exploración de un video
  cualquiera con muestreo determinista de 16 frames via `np.linspace`, grilla
  4x4 con matplotlib, impresión de metadatos (fps, duración, resolución).

Fase de pipeline (día 3, en pareja con Diego, que dejó `extract_frames`,
`normalize_frames`, `to_tensor`, constantes y esqueleto de la clase):

- **Rama `feat/preprocess-completar-pipeline`:** completo los stubs
  `NotImplementedError` que Diego dejó en `src/preprocessing/video_pipeline.py`.
  - `resize_and_convert`: `cv2.resize` con `INTER_AREA` (menos aliasing al
    achicar) + `cvtColor(BGR→RGB)`. Mantiene dtype `uint8` para que
    `normalize_frames` arranque correctamente desde ahí.
  - `VideoPreprocessor.__call__`: encadena extract → resize/convert →
    normalize → to_tensor, llamando a `augment_frames` sólo cuando `train=True`
    (Grageda rellena ese stub en la fase 4).
  - `src/preprocessing/__init__.py`: expone la clase y las funciones públicas.
- **Suite de tests `tests/test_preprocessing.py` (10 tests):** cubre cada paso
  por separado y la integración completa — shape, dtype, rangos, conversión
  BGR→RGB con pixel conocido, estadísticas ImageNet exactas con frame gris,
  video más corto que `n_frames`, archivo faltante, video corrupto. 9 pasan
  en verde sobre fixtures sintéticos; 1 opcional (`HOCKEY_VIDEO_PATH`) salta
  si no se setea la env var.
- **Fixtures sintéticos (`tests/fixtures/make_fixture.py`):** genera
  `sample_video.mp4` (90 frames, 320x240), `short_video.mp4` (10 frames) y
  `video_corrupto.mp4` (bytes aleatorios con semilla fija para
  reproducibilidad). Los tres `.mp4` se commitean al repo (~84 KB en total)
  para que los tests corran sin descargar datasets.
- **`pytest.ini`:** `pythonpath = .` para que los tests resuelvan
  `from src.preprocessing import ...` desde la raíz del repo.
- **Validación manual** sobre un `.avi` real de RWF-2000 y un `.mp4` de
  Real Life Violence: ambos devuelven shape `(16, 3, 224, 224)`, `float32`,
  rango `[-2.12, +2.64]` — tocan los extremos teóricos de ImageNet.

Pendiente de mi parte según el plan: entrenamiento (con Grageda) y `model.py`
(con Lorgio). Se registrará aquí cuando se haga.

## P2 — Lorgio

- (pendiente)

## P3 — Diego

Fase de datos (en pareja con Grageda, que se encargó de Real Life Violence):

- **RWF-2000:** descarga completa desde Zenodo
  (https://zenodo.org/records/15687512). Verificación local: los 13 archivos
  coinciden con los MD5 publicados, 2000 clips (800 + 800 en `train`,
  200 + 200 en `val`), sin archivos vacíos. Clips MJPEG de 5 s a 30 fps.
- **Nombres demasiado largos:** 24 videos de `Train_Fight` tenían nombres de
  más de 255 bytes, que Linux no admite. Se renombraron como
  `longname_NN_urlgot_XXX.avi`; el contenido es el original.
- **Subida al Drive del grupo** en `datasets/rwf2000/RWF-2000/`, conservando
  la estructura real (`train/` y `val/`).
- **Hockey Fight:** descargado de Kaggle (1000 clips `.avi`, 500 pelea y 500
  no pelea) para probar el pipeline antes de usar RWF-2000.
- Los videos no se versionan; viven en el Drive y en mi copia local.

Pendiente de mi parte según el plan: `video_pipeline.py` (con Ariany),
`model.py` (con Lorgio) y frontend (con Grageda). Se registrará aquí cuando
se haga.

## P4 — Grageda

Fase de datos (en pareja con Diego, que se encargó de RWF-2000):

- **Real Life Violence Situations:** descarga desde Kaggle
  (`mohamedmustafa/real-life-violence-situations-dataset`) con mi cuenta.
  Verificación local: 2000 videos (1000 `Violence` + 1000 `NonViolence`),
  1.81 GB, sin archivos vacíos ni ilegibles al abrirlos con OpenCV. El zip
  de 4 GB traía el dataset duplicado (MD5 idénticos), así que se subió una
  sola copia.
- **Subida al Drive del grupo** en `datasets/real_life/`, separada de
  `rwf2000/`. Conteo verificado en el Drive: 1000 y 1000.
- **Documentación de datasets:** reescribí `data/README.md` con las fuentes,
  licencias y estructura real de RWF-2000, Real Life y Hockey Fight
  (commit `7ae8319`, PR #2).
- Los videos y el manifiesto de verificación no se versionan; viven fuera
  del repo (Drive / copia local).

Fase de EDA (rama `feat/eda-dataset`):

- **`src/preprocessing/dataset_metadata.py`** (commit `d85c5de`): lee los metadatos de los videos
  (duración, fps, resolución, bytes, clase y split deducidos de la carpeta)
  sin decodificar el clip completo. Tests en `tests/test_dataset_metadata.py`.
- **`notebooks/02_eda_datasets.ipynb`** (commit `a5e1e4e`): EDA de Real Life (balance, integridad,
  duración, fps, resolución, muestra visual). Hallazgo: la clase se predice solo
  con la resolución (77,5 %) o solo con fps (81,7 %), un posible atajo para el
  modelo. La sección de RWF-2000 está lista pero no se ejecutó: falta tener esos
  videos en `data/raw/rwf2000/`.
- **`notebooks/03_comparar_frames.ipynb`** (commit `e7990b5`): comparación 8 vs 16 vs 32 frames con
  80 clips de Real Life: separación temporal, redundancia y costo (extracción,
  memoria, forward de MobileNetV2). No mide exactitud: eso queda para la fase
  de entrenamiento.
- Se agregó `pandas` a `requirements.txt` (commit `d85c5de`).
- Lo hice yo solo: la parte de Lorgio (revisar y complementar el EDA) queda
  pendiente y se registrará en su sección si la hace.

Pendiente de mi parte según el plan: ejecutar el EDA con RWF-2000, entrenamiento
(con Ariany) y frontend (con Diego). Se registrará aquí cuando se haga.
