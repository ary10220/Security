# Frontend (Vue 3 + Vite)

Dashboard con cinco pestañas: Analizar, Lote, Métricas, Datos e Historial. La API FastAPI debe estar iniciada en `localhost:8000`.

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
| `VITE_API_URL` | `http://localhost:8000` | URL del backend FastAPI |

El dashboard actual obtiene predicciones, métricas y frames desde la API. El historial guarda solo los últimos 20 nombres, fechas y resultados en `localStorage`; no guarda videos.
La pestaña Analizar también permite grabar 10 segundos con webcam si el navegador puede producir MP4. Los frames del modal de preprocesamiento provienen de `GET /preprocessing_demo` usando el `video_id` de `/predict`.

## Contrato con la API

El cliente del dashboard está en `src/composables/useApi.js`.

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

`GET /metrics?model=...`: lee el `metrics_<modelo>.json` de test y añade `model_size_mb` si existe el checkpoint. La pestaña ofrece solo modelos que ya tienen métricas disponibles.

`POST /predict/batch` (multipart, campo repetido `files`): acepta hasta 20 videos y devuelve un resultado o un error por archivo.

`GET /health` informa si hay un checkpoint cargado. `GET /preprocessing_demo?video_id=...` devuelve los 16 frames del video analizado.

## Pendiente (no definido por el plan o el equipo)

- Cortes del nivel de alerta (`ALERT_LEVELS` en `src/config.js`, provisionales).
- Duración del segmento (2 s o 3 s).
- Tamaño máximo de subida (`MAX_UPLOAD_MB`), que debe coincidir con el backend.
- La webcam depende de que el navegador permita grabar MP4; si no, se puede subir un archivo.
