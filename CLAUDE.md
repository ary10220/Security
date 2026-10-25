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