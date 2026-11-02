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

Fase de exploración (día 2, PR #5, commit `996bbe8`):

- **`notebooks/01_explore_dataset.ipynb`:** lo ejecuté con un clip real de
  RWF-2000 (`_2RYnSFPD_U_0.avi`: 1280x720, 30 fps, 150 frames, 5 s) y dejé las
  salidas guardadas. Agregué la cita del dataset y la validación
  `cap.isOpened()`. El muestreo de 16 frames y la grilla son de Ariany (commit
  `143fa3f`).

Fase de modelo (día 5, PR #17, en pareja con Diego, que dejó `ViolenceDetector`,
los stubs y los tests):

- **`src/models/model.py`:**
  - Clases base `_FrameBackbone` y `_LSTMDetector` (commit `82ba657`): moví ahí
    el cuerpo de `ViolenceDetector`, como sugería el TODO de Diego. Por fuera no
    cambia: mismas claves del `state_dict` y salida idéntica con los mismos pesos.
  - `FrameBaseline` (commit `d0d813b`): el mismo MobileNetV2 congelado y una capa
    lineal sobre el frame del medio, para medir si la LSTM aporta.
  - `ResNetDetector` (commit `dc9d216`): ResNet50 sin `avgpool` ni `fc`, con la
    misma LSTM y la misma cabeza. `build_model` crea los 3 modelos.
  - `save_checkpoint` y `load_checkpoint` (commit `81fd9c1`): guardan el nombre
    del modelo junto a los pesos. Cargan con `map_location="cpu"` y
    `weights_only=True`.
- **`src/models/train.py`:** guarda el mejor epoch con `save_checkpoint`.
- **`tests/test_model.py`:** quité los skips de mi parte y agregué 2 tests de
  checkpoints. También activé la prueba con pesos de ImageNet (commit `88f0de8`).
- **Verificación con pesos reales:** los 3 modelos dan logits `(1, 2)` finitos
  sobre un clip de RWF-2000. El smoke test de `train.py` corre con los 3 y sus
  checkpoints se recargan como la clase correcta.
- **`DEFAULT_HYPERPARAMS`:** se mantienen sin cambios. Las métricas de
  `artifacts/` usan los valores de `scripts/train_local_cpu.py`.

Revisión del EDA (`notebooks/02_eda_datasets.ipynb` y
`notebooks/03_comparar_frames.ipynb`, de Grageda):

- **Alcance:** la revisión la hizo Claude Code a pedido mío: leyó el código, las
  conclusiones y las salidas guardadas, y corrió los tests. Los notebooks no se
  volvieron a ejecutar porque no tengo los videos.
- **Sin problemas:** `threshold_shortcut` elige el umbral en un conjunto y lo
  evalúa en otro. En el notebook 02, los conteos, las duraciones, las
  resoluciones, los atajos y las fuentes de las conclusiones coinciden con las
  salidas impresas. El rango de fps de Real Life solo se ve en una figura, así
  que no se pudo confirmar. No hay rutas absolutas guardadas.
- **Detalles a corregir:**
  - La sección de Grageda dice fps 81,7 %; el notebook 02 mide 81,3 %.
  - El notebook 03 dice que pasar de 16 a 32 frames multiplica el costo por ~2 y
    que crece linealmente. Su salida da 2,46 veces. La recomendación de usar 16
    no cambia.
- **Complemento:** los modelos no superan a la regla de metadatos. `alto > 251`
  acierta 75 % en los 400 clips de `val/`; `mobilenet_lstm` acierta 64 % y
  `mobilenet_baseline` 66 % en 100 clips. Los clips no son los mismos y los
  intervalos al 95 % se solapan (54-73 % y 56-75 % contra 71-79 %), así que no
  prueba que sean peores, pero tampoco hay evidencia de que sean mejores.

Hecho con asistencia de Claude Code.

Pendiente de mi parte: entrenar y evaluar `resnet_lstm` con los mismos clips que
los otros dos modelos (la copia de Ariany, de 1348 y 333 clips).

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

Fase de integración (1 nov):

- **Conversión de videos:** el frontend solo acepta `.mp4` y RWF-2000 viene
  en `.avi`, así que convertí 972 de los 2000 clips con `ffmpeg` (H.264,
  `yuv420p`, sin audio): los 800 de `Train_Fight` y 172 de `Train_NonFight`.
  Los `.avi` originales no se tocaron y los `.mp4` quedaron fuera del repo.
- **Script `scripts/eval_test_por_clip.py` (commit `fc02cc2`, rama
  `feat/eval-test-por-clip`):** reconstruye el mismo test de
  `train_local_cpu.py` (semilla 42) y guarda, por clip, el video, la etiqueta
  real, la predicción y el score. Sirve para identificar qué clips falla el
  modelo en el informe. No entrena ni modifica el checkpoint.
- **Hallazgo sobre los datos:** el log de la corrida de Ariany indica 1348
  clips en `train/` y 333 en `val/`; mi copia tiene 1600 y 400. Con listas
  distintas, la semilla 42 no muestrea los mismos 100 clips de test, así que
  mi evaluación no es comparable clip a clip con la suya.
- **Prueba de integración con la pestaña Lote** (API y frontend locales, modelo
  `mobilenet_lstm` en `artifacts/`): 30 clips (3 lotes de 10) dieron 22
  aciertos (73 %), y 100 clips (5 lotes de 20, 10 peleas y 10 normales cada
  uno, de `train/`) dieron **64 %**: 36 de 50 peleas detectadas (72 %) y 22 de
  50 normales marcados como sospechosos (44 %). La matriz,
  `[[28, 22], [14, 36]]`, coincide con la publicada por Ariany. Con 100 clips
  el margen de error ronda ±5 puntos (los lotes van de 50 % a 85 %), por lo
  que esa coincidencia no prueba que sea el mismo modelo.
- **Procedencia del `.pt`:** el `model_mobilenet_lstm.pt` que usé estaba en mi
  carpeta `artifacts/` y no sé quién lo entrenó. Sus resultados son
  consistentes con los de Ariany, pero no está confirmado que sea su
  checkpoint.
- **Comportamiento observado:** el modelo casi no deja pasar peleas, pero
  marca como sospechosa casi la mitad de los videos normales. Un score por
  encima de 0,5 no basta: la API alerta solo con más de 3 s consecutivos
  sobre el umbral, y por eso hay peleas con 55-61 % que salen NORMAL.
  Ejemplos: falso negativo `s-jW4cyavW0_0` (61,1 %); falsos positivos
  `API3q8TBGu4_0` (70,7 %) y `36zWWHlldS8_0` (69,5 %).
- Hecho con asistencia de Claude Code (declarar en el informe).

Pendiente de mi parte: los 3 ejemplos de error del informe, que deben salir
del checkpoint y del dataset de Ariany (corriendo `eval_test_por_clip.py` en
su máquina) para que cuadren con sus cifras; abrir el PR de la rama
`feat/eval-test-por-clip`; reporte PDF del timeline (opcional, el clip
depende de `clip_url`, que la API aún no implementa).

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
  con la resolución (77,5 %) o solo con fps (81,3 %), un posible atajo para el
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

Fase de correcciones tras la revisión del EDA (rama `fix/eda-cifras-grageda`):

- Lorgio revisó los notebooks 02 y 03 (PR #28) y señaló dos cifras mías mal
  copiadas. Las comprobé contra las salidas guardadas y las corregí:
  - `CONTRIBUCIONES.md`: fps de Real Life 81,7 % → 81,3 % (el notebook 02 imprime
    0,813).
  - `notebooks/03_comparar_frames.ipynb`: pasar de 16 a 32 frames multiplica el
    tiempo de MobileNetV2 por ~2,5 (2,46 veces: 168,5 → 415,2 ms), no por ~2, y no
    crece linealmente (de 8 a 16 sube 2,35 veces). También corregí la misma
    imprecisión en la recomendación final. La recomendación de usar 16 frames no
    cambia.
- Solo cambié texto: los notebooks no se reejecutaron y sus salidas guardadas son
  las mismas.
- Comprobé el rango de fps de Real Life que Lorgio no pudo confirmar (10–37, según
  la conclusión del notebook 02) con `data/processed/metadata_real_life.csv`: va
  de 10,5 a 37,0 fps. Es consistente con lo escrito.
- Hecho con asistencia de Claude Code.

Pendiente de mi parte: integración del frontend con la API real y revisión del
trabajo conjunto con Diego.
