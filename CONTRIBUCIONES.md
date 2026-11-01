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

Fase de entrenamiento (día 6, rama `feat/model-train-loop`, en pareja con
Grageda que se suma al bucle de entrenamiento):

- **`src/models/dataset.py`** (`VideoDataset`): lee carpetas con la
  estructura `<split>/<clase>/*.{mp4,avi}` (RWF-2000, Real Life, Hockey) y
  autodetecta la etiqueta por el nombre de la carpeta (`fight`/`violence` → 1,
  prefijo `non` → 0). Si un video no se puede abrir se loguea y se intenta
  con otro vecino aleatorio hasta `MAX_RETRIES`, para que un `.mp4` corrupto
  en 2000 no caiga el entrenamiento. Admite un `label_map` explícito para
  nomenclaturas raras.
- **`src/models/metrics.py`** (`compute_metrics`): accuracy, F1 macro, F1 /
  precision / recall por clase y matriz de confusión 2x2 con `sklearn`. F1
  macro es la métrica que elige el mejor checkpoint (con clases
  desbalanceadas, accuracy sube siempre pero F1 macro baja si el modelo
  ignora la clase chica).
- **`src/models/train.py`**: script con `argparse` que llama
  `build_model(args.model)` del `model.py` de Diego, por lo que ya soporta
  los 3 modelos aunque los stubs de Lorgio (`resnet_lstm`,
  `mobilenet_baseline`) todavía lanzan `NotImplementedError` — el script lo
  captura y sale con warning. Loop train + val con `tqdm`, Adam sobre
  params con `requires_grad=True` (backbone congelado), `ReduceLROnPlateau`
  sobre F1 macro, guardado del mejor checkpoint y JSON de métricas con
  historia por época. `--smoke-test` entrena 1 época con 10+5 videos
  estratificados. `torch.save(state_dict)` directo hasta que Lorgio
  implemente `save_checkpoint`.
- **`scripts/make_mini_dataset.py`:** generador sintético de 16 `.mp4`
  (`data/raw/mini/train/{Fight,NonFight}` y `.../val/...`) para correr el
  smoke test sin bajar un dataset real. Los videos van bajo `data/raw/` por
  lo que no se commitean.
- **`requirements.txt`:** agregué `scikit-learn==1.5.0`, `scipy==1.13.1`
  (clave: scipy ≥ 1.14 exige numpy ≥ 2 y rompería `torch==2.3.1`) y
  `tqdm==4.66.4`. También `.gitignore` para `artifacts/*.json`.
- **Smoke test en CPU** con `mobilenet_lstm`: 1 época, 10 train + 5 val,
  6.7 s. Checkpoint de 14.3 MB y `metrics_mobilenet_lstm.json` escritos
  correctamente. Las métricas son aleatorias como se esperaba (videos
  sintéticos), lo importante es que DataLoader, forward, backward y
  guardado funcionan de punta a punta.

Pendiente de mi parte según el plan: entrenamiento serio con RWF-2000
(epochs=15, dataset completo, GPU si consigo); frontend queda para la fase
final. Se registrará aquí cuando se haga.

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

Fase de pipeline (días 3-4, en pareja con Ariany, que completó
`resize_and_convert`, el encadenado de `VideoPreprocessor` y la suite base):

- **Rama `feat/preprocess-frames` (PR #6):** esqueleto de
  `src/preprocessing/video_pipeline.py` con constantes y stubs (commit
  `8c891ab`), `extract_frames` con 16 frames uniformes vía OpenCV y relleno
  con el último frame si el video es corto (`394939f`), `normalize_frames` con
  media/std de ImageNet (`01d9257`) y `to_tensor` a `(T, C, H, W)` (`f649286`).
- **Rama `feat/preprocess-augment` (PR #11):** `augment_frames` (paso 3.4, commit
  `d4b70aa`). Flip horizontal (p=0.5), recorte aleatorio del 80-100 % del área
  reescalado al tamaño original y jitter leve de brillo/contraste (±20 %),
  sorteados una sola vez por clip para no romper la coherencia temporal. El
  jitter se aplica en espacio de píxeles `[0, 1]` y se vuelve a normalizar.
  Acepta un `torch.Generator` para resultados reproducibles y funciona con
  clips en CPU o GPU (commit `c6b3a0e`). 8 tests nuevos en
  `tests/test_preprocessing.py` (forma, mismo cambio en todos los frames,
  reproducibilidad, no modifica la entrada, rango ImageNet, GPU y pipeline con
  `train=True` comparado contra la salida sin augmentation). Validado sobre un
  `.avi` real de Hockey Fight.

Fase de modelo (día 5, en pareja con Lorgio):

- **Rama `feat/model-violence-detector` (PR #12):** `ViolenceDetector` en
  `src/models/model.py` (MobileNetV2 congelado + BiLSTM, commit `2963dee`) con
  8 tests en `tests/test_model.py`. Usa el estado final `h_n` de la BiLSTM y no
  `out[:, -1]` (en ese paso la dirección hacia atrás solo vio un frame), y
  `train()` deja el backbone congelado en `eval()` para que BatchNorm no
  cambie sus estadísticas.
- **Para Lorgio** (commit `8d308ef`): dejé la estructura y los stubs marcados
  con `TODO(Lorgio)` en `src/models/model.py`: `FrameBaseline`, `ResNetDetector`, `build_model`
  (ya despacha los 3 nombres de `train.py --model`), `save_checkpoint`,
  `load_checkpoint` y confirmar `DEFAULT_HYPERPARAMS`. Sus tests ya están
  escritos en `tests/test_model.py` con `skip`: al implementar cada pieza se
  quita el `skip`. Ariany y Grageda pueden empezar `train.py` ya con
  `mobilenet_lstm`.
- **Importante para Lorgio:** los tests usan `pretrained=False` (sin internet),
  pero nada se ha probado todavía con los pesos reales de ImageNet. Hay que
  correr los 3 modelos con `pretrained=True` sobre un clip real (test
  `test_pesos_imagenet_con_video_real`, con `HOCKEY_VIDEO_PATH`) y comprobar
  logits finitos y checkpoint que recarga igual.

Fase de frontend (en pareja con Grageda, que armó la interfaz en el PR #15):

- **Revisión y corrección (PR #18, commit `8ed39bd`):** revisé la interfaz de
  Grageda y corregí un fallo del video en `frontend/src/App.vue` (evitar
  `null` en `timeupdate`).
- **Frontend contra la API real (PR #22, commits `a856b99` y `578afa6`):**
  quité los datos simulados (`mock.js`, `frames.js` y la variable
  `VITE_USE_MOCK`). `frontend/src/lib/api.js` llama siempre al backend real y
  la vista de preprocesamiento pide los 16 frames a `/preprocessing_demo` con
  el `video_id` que devuelve `/predict`, para no reimplementar el
  preprocesamiento en el navegador. Actualicé el README del frontend.
- **Webcam (PR #23, commits `e1bca1e` y `067f0ca`):** botón "Usar webcam"
  (`WebcamRecorder.vue` y `lib/webcam.js`) que graba 10 s con `MediaRecorder`
  y envía el archivo al mismo análisis que la carga de video. Solo graba en
  mp4, porque la API acepta mp4, avi y mov y no webm; en Firefox, que solo
  graba webm, la interfaz muestra un aviso. 5 tests en
  `frontend/src/lib/webcam.test.js`.
- Hecho con asistencia de Claude Code (declarar en el informe).

Pendiente de mi parte: prueba de punta a punta con la API y el modelo
entrenado (`.pt`) cuando el equipo lo comparta; reporte PDF del timeline
(opcional, el clip depende de `clip_url`, que la API aún no implementa).

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

Fase de EDA con RWF-2000 (rama `feat/eda-rwf2000`, apilada sobre `feat/eda-dataset`):

- Bajé RWF-2000 del Drive (6 `.zip`), verifiqué integridad (CRC) y conteos
  (800/800/200/200, 2000 videos únicos), y lo ordené en `data/raw/rwf2000/`.
- **`src/preprocessing/dataset_metadata.py`** (commit `8f5b891`): agregué
  `threshold_shortcut`, que mide cuánto separa las clases una sola característica
  con un umbral elegido en un conjunto y evaluado en otro. Tests nuevos en
  `tests/test_dataset_metadata.py`.
- **`notebooks/02_eda_datasets.ipynb`** (commit `b2b84e4`): ahora cubre los dos
  datasets. Hallazgos medidos: RWF-2000 es uniforme en duración y fps (5 s, 30
  fps) pero no en resolución; la resolución lo separa ~75 % (atajo posible);
  los clips salen de ~961 videos fuente y ninguno está en train y val a la vez.
- **`notebooks/03_comparar_frames.ipynb`** (commit `b1b6ce2`): la comparación
  8/16/32 se repitió con 160 clips (Real Life y RWF-2000); la conclusión de
  mantener 16 se sostiene en ambos en costo y cobertura (no en exactitud).
- **Documentación** (commit de docs de esta rama): estado del proyecto,
  hallazgos y decisiones abiertas en `CLAUDE.md`; cómo dejar los datos en
  `data/README.md`; nota de versión de Python en `README.md`.
- Sigue pendiente la parte de Lorgio (revisar y complementar el EDA).

Fase de entrenamiento (rama `feat/pr-curve`, parte de la pareja con Ariany):

- **`src/models/metrics.py`** (commit `40ffb4f`): `pr_curve(y_true, y_score)`
  devuelve precisión, recall y umbrales de la clase sospechosa, el average
  precision y el umbral que maximiza F1. Recorta la curva a `max_points` para
  que el JSON no crezca. Tests en `tests/test_metrics.py`, con datos sintéticos.
- **`src/models/train.py`** (rama `feat/train-save-scores`, commit `36ae908`,
  propuesta para revisión de Ariany, autora del archivo): `_run_epoch` ahora
  devuelve también la probabilidad de la clase sospechosa y `metrics_*.json`
  guarda la curva PR del mejor epoch (`pr_curve`, `best_epoch`). Probado con el
  smoke test sintético; la curva de ese run no significa nada.

Fase de frontend (rama `feat/frontend-vue`, en pareja con Diego):

- **`frontend/`:** scaffold Vue 3 + Vite y la interfaz del plan: carga
  drag-and-drop de `.mp4` con validación de tipo y tamaño, veredicto con nivel
  de alerta y momento más crítico, timeline interactivo (clic salta al segundo
  del video), slider de umbral que recalcula el veredicto en el navegador,
  panel de métricas (por clase, matriz de confusión y curva PR) y vista de
  preprocesamiento (16 frames).
- La lógica (`frontend/src/lib/analysis.js`) tiene 7 tests con vitest.
- **Sin backend todavía:** la app corre con datos simulados, rotulados como tales
  en la interfaz. `frontend/README.md` documenta el contrato propuesto con la
  API, pendiente de confirmar con quien haga `api/`.
- Hecho con asistencia de Claude Code (declarar en el informe).
- Queda sin implementar: vista de `/report`, modalidad webcam (opcional en el
  plan) e integración con el backend real.

Pendiente de mi parte: integración del frontend con la API real y revisión del
trabajo conjunto con Diego.
