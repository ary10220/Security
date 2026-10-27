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

## Estado del proyecto (actualizado el 27 oct 2026)

Resumen para quien retome el trabajo con Claude. Es un estado, no un plan:
si algo cambia, actualizar esta sección en el mismo PR.

### Hecho
- **Datos:** RWF-2000 (2000 clips, Zenodo, CC BY 4.0) y Real Life Violence
  (2000 clips, Kaggle) están en el Drive del grupo en `datasets/`. Hockey Fight
  fue descargado por Diego. Detalle, licencias y estructura real en
  [`data/README.md`](data/README.md).
- **Preprocesamiento:** `src/preprocessing/video_pipeline.py` implementa los
  pasos 1, 2, 3, 5 y 6 (extraer 16 frames, resize+RGB, normalizar, tensor
  `(16, 3, 224, 224)`, clase `VideoPreprocessor`), con tests en
  `tests/test_preprocessing.py`.
- **EDA (Grageda):** `src/preprocessing/dataset_metadata.py` (metadatos de
  videos y `threshold_shortcut`), `notebooks/02_eda_datasets.ipynb` y
  `notebooks/03_comparar_frames.ipynb`. La parte de Lorgio (revisar y
  complementar) sigue pendiente.

### Pendiente
- `augment_frames` (paso 3.4) sigue como stub (`NotImplementedError`). Hay
  contradicción sobre quién lo hace (el plan dice Ariany + Diego; una nota en
  `CONTRIBUCIONES.md` dice Grageda): confirmar antes de implementarlo.
- `src/models/` (modelo MobileNetV2 + BiLSTM), entrenamiento (`train.py`,
  métricas, matriz de confusión, curva PR), `api/` y `frontend/`: sin empezar.
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
4. ¿Quién implementa `augment_frames`?
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