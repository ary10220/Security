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
│   ├── rwf2000/
│   ├── real_life/
│   └── hockey/   # solo si se usa para pruebas iniciales
└── processed/    # Frames y tensores precomputados. NO commitear.
```

La copia compartida del grupo vive en el Drive, en `datasets/` (con las
mismas subcarpetas `rwf2000/` y `real_life/`). Es una ubicación distinta de
`data/raw/`, que es la copia local de cada integrante.

### `raw/`

Videos originales tal cual se descargaron de la fuente. **No modificar
estos archivos**: el pipeline de preprocesamiento siempre arranca desde
aquí, por lo que si se corrompen o se editan a mano, los resultados dejan
de ser reproducibles.

### `processed/`

Salida del pipeline de extracción de frames (ver `src/preprocessing/`).
Contiene los 16 frames muestreados por video, redimensionados a 224x224
(entrada que espera MobileNetV2), serializados en `.npy` o `.pt`.

Esta carpeta se regenera corriendo el pipeline, no se versiona.

## Fuentes de datos

### RWF-2000 (dataset principal)

- **Contenido:** 2000 clips de 5 s de cámaras de vigilancia.
- **Fuente de descarga usada:** Zenodo, https://zenodo.org/records/15687512
  (registro a nombre de Ming Cheng, Duke Kunshan University, uno de los
  autores). El repo de GitHub original no incluye los videos y el
  formulario de solicitud no se responde.
- **Licencia del registro de Zenodo:** CC BY 4.0.
- **Tamaño real:** ~9.4 GB comprimido, ~12 GB extraído (no ~2 GB).
- **Verificación:** los 13 archivos coinciden con los MD5 publicados; no hay
  videos corruptos ni vacíos.
- **Estructura real** (distinta de `Fight/NonFight` que asumía el plan):

  | Carpeta | Videos |
  |---|---|
  | `train/Train_Fight` | 800 |
  | `train/Train_NonFight` | 800 |
  | `val/Val_Fight` | 200 |
  | `val/Val_NonFight` | 200 |

- **Contenido medido** (EDA en
  [`notebooks/02_eda_datasets.ipynb`](../notebooks/02_eda_datasets.ipynb), sobre
  la copia completa de 2000 videos bajada del Drive):
  - Todos los clips son `.avi` de **5,0 s a 30 fps**; 0 ilegibles.
  - La resolución **no es uniforme**: 700 clips de 320x240, 628 de 1280x720,
    31 de 1920x1080 y muchas otras (4:3, 16:9, verticales).
  - La resolución separa las clases con ~75 % de exactitud (los clips de
    pelea tienden a ser más grandes), así que también hay un posible atajo
    aquí. El fps y la duración no delatan la clase (50 %).
  - Los 2000 clips salen de ~961 videos fuente (aproximado por el prefijo del
    nombre `<id>_<n>.avi`). Ninguna fuente aparece en `train` y `val` a la
    vez. **Si se hace otro split o validación cruzada, dividir por fuente, no
    por clip.**
- **Nota:** 24 videos de `Train_Fight` tenían nombres en chino de más de
  255 bytes, que Linux no admite. Se renombraron como
  `longname_NN_urlgot_XXX.avi`; el contenido es el original.
- **Cita:** Cheng et al., "RWF-2000: An Open Large Scale Video Database for
  Violence Detection", ICPR 2020.

### Real Life Violence Situations (complemento de diversidad)

- **Fuente:** Kaggle,
  https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
  (autor del dataset en Kaggle: Mohamed Mustafa).
- **Licencia mostrada por Kaggle:** "Data files © Original Authors". No es
  una licencia abierta: uso académico con cita obligatoria. No
  redistribuir fuera del grupo.
- **Cita solicitada por los autores:** M. Soliman, M. Kamal, M. Nashed,
  Y. Mostafa, B. Chawky, D. Khattab, "Violence Recognition from Videos using
  Deep Learning Techniques", Proc. 9th International Conference on
  Intelligent Computing and Information Systems (ICICIS'19), El Cairo,
  pp. 79-84, 2019.
- **Contenido verificado** (descarga del 25/10/2026):

  | Carpeta | Videos | Tamaño | Formato |
  |---|---|---|---|
  | `Violence` | 1000 | 1.01 GB | 1000 `.mp4` |
  | `NonViolence` | 1000 | 0.80 GB | 951 `.mp4` + 49 `.avi` |

  Total: 2000 videos, 1.81 GB. Todos se abren y se leen con OpenCV; ninguno
  está vacío.
- **El zip de Kaggle pesa 4 GB** porque trae el dataset **duplicado** (la
  misma estructura aparece dos veces, con MD5 idénticos). Solo se subió una
  copia: la carpeta `Real Life Violence Dataset`.
- **Los videos no son uniformes:** la duración mediana es 5 s, pero hay
  clips desde 1 s hasta ~376 s; las resoluciones van desde 136x240 hasta
  1080x1920 y los fps varían. El preprocesador (16 frames uniformes,
  224x224) debe tolerarlo; el análisis está en
  [`notebooks/02_eda_datasets.ipynb`](../notebooks/02_eda_datasets.ipynb).
- **Hallazgos del EDA** (medidos sobre estos 2000 videos):
  - Solo 3 clips duran más de 30 s (`V_789` ≈ 376 s, `NV_992` ≈ 180 s,
    `V_398` ≈ 137 s). Con 16 frames uniformes quedarían muy separados;
    pendiente decidir si se excluyen o se recortan.
  - Las clases difieren en características que no son el comportamiento:
    solo con la resolución (224x224 → no violencia) se acierta el 77,5 % y
    solo con fps > 29,4 (umbral elegido en una mitad y evaluado en la otra),
    el 81,3 %. Por eso conviene evaluar también en RWF-2000 y reportar ambos
    resultados por separado.
  - No trae split ni identificador de fuente: una división al azar puede
    mezclar clips muy parecidos entre train y test.
  - La muestra incluye deportes de contacto y tomas de transmisión, no solo
    cámaras de vigilancia.
- **Estructura prevista en local:** `data/raw/real_life/Violence/` y
  `data/raw/real_life/NonViolence/`.

### Hockey Fight (pruebas iniciales del pipeline)

- **Contenido:** ~1000 clips cortos de peleas en partidos de hockey
  (~500 MB según el plan; confirmar al descargarlo).
- **Uso:** conjunto pequeño para probar el pipeline, no sustituye a los
  datasets principales.
- **Cita:** Nievas et al., 2011.

## Cómo dejar los datos en `data/raw/`

Descargar `datasets/rwf2000/` y `datasets/real_life/` del Drive (si Drive lo
entrega en varios `.zip` de ~2 GB, bajarlos **todos**). Extraer de modo que
queden estas rutas, sin carpetas intermedias de más:

```
data/raw/rwf2000/train/Train_Fight/*.avi      (800)
data/raw/rwf2000/train/Train_NonFight/*.avi   (800)
data/raw/rwf2000/val/Val_Fight/*.avi          (200)
data/raw/rwf2000/val/Val_NonFight/*.avi       (200)
data/raw/real_life/Violence/*.mp4             (1000)
data/raw/real_life/NonViolence/*.mp4 y .avi   (1000)
```

Comprobación rápida (PowerShell), debe dar 800/800/200/200:

```powershell
Get-ChildItem data\raw\rwf2000 -Recurse -File | Group-Object { $_.Directory.Name } | Select Name, Count
```

El notebook `02_eda_datasets.ipynb` escanea los videos y guarda la tabla de
metadatos en `data/processed/metadata_*.csv` (tampoco se versiona); el
notebook 03 la reutiliza.

## Cómo descargar

Pendiente: el plan propone un script `data/download_data.sh` que copie los
datos desde el Drive del grupo con `gdown`. Mientras tanto, pedir acceso a la
carpeta `datasets/` del Drive y copiar cada dataset a su subcarpeta de
`data/raw/`.

No subir videos al repositorio ni credenciales de Kaggle (`kaggle.json`,
tokens) a ningún lugar compartido.
