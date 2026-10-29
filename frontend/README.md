# Frontend (Vue 3 + Vite)

Interfaz web para subir un video, ver el veredicto, la línea de tiempo, ajustar el umbral y consultar las métricas del modelo y el preprocesamiento.

## Cómo correrlo

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm test         # lógica del análisis (vitest)
npm run build
```

Variables (`.env.local`, no se versiona):

| Variable | Por defecto | Efecto |
|---|---|---|
| `VITE_USE_MOCK` | `true` | `true` usa datos simulados; `false` llama al backend |
| `VITE_API_URL` | `http://localhost:8000` | URL del backend FastAPI |

Con `VITE_USE_MOCK=true` la interfaz lo avisa con un banner: el timeline y las métricas son **simulados**, no resultados del modelo. Los frames del tab de preprocesamiento sí salen del video subido, pero se calculan en el navegador (muestreo uniforme + resize), no con el pipeline real.

## Contrato propuesto con la API

Es una **propuesta del frontend**: hay que confirmarla con quien implemente `api/`. Si el backend responde distinto, solo se ajusta `src/lib/api.js`.

`POST /predict` (multipart, campo `file`):

```json
{
  "video": { "name": "clip.mp4", "duration_s": 20.0 },
  "segment_seconds": 2,
  "model": "mobilenet_lstm",
  "default_threshold": 0.5,
  "segments": [{ "start": 0, "end": 2, "score": 0.12 }],
  "clip_url": "http://.../clip.mp4"
}
```

- `score` es la probabilidad de la clase "sospechoso" (softmax), no la etiqueta. El umbral se aplica en el navegador, por eso el slider no llama al backend.
- `segment_seconds` lo manda la API: el plan no fija 2 s o 3 s.
- `clip_url` puede ser `null`; el botón de descarga queda deshabilitado.

`GET /metrics`: acepta el `metrics_<modelo>.json` de `train.py` (`{ "model", "final": {...} }`) o el formato plano. Campos usados: `class_names`, `accuracy`, `f1_macro`, `f1_per_class`, `precision_per_class`, `recall_per_class`, `confusion_matrix`, `inference_ms` y `pr_curve` (la que devuelve `pr_curve()` de `src/models/metrics.py`).

`POST /preprocess` (multipart): `{ "n_frames": 16, "size": [224, 224], "frames": [{ "index": 0, "time": 0.1, "src": "data:image/jpeg;base64,..." }] }`.

`GET /health` y `GET /report` también están en el cliente; `/report` aún no tiene vista.

## Pendiente (no definido por el plan o el equipo)

- Cortes del nivel de alerta (`ALERT_LEVELS` en `src/config.js`, provisionales).
- Duración del segmento (2 s o 3 s).
- Tamaño máximo de subida (`MAX_UPLOAD_MB`), que debe coincidir con el backend.
- Modalidad webcam: el plan la deja como opcional y no está implementada.
