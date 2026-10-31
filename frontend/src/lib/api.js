// Cliente del backend FastAPI (api/main.py). Todo viene del backend real: el
// frontend no genera resultados por su cuenta.
import { API_URL } from '../config.js'

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_URL}${path}`, options)
  } catch {
    throw new Error(`No se pudo conectar con el backend (${API_URL}). ¿Está encendido?`)
  }
  if (!response.ok) {
    let detail = ''
    try {
      detail = (await response.json()).detail ?? ''
    } catch {
      /* respuesta sin JSON */
    }
    throw new Error(detail || `El backend respondió ${response.status}`)
  }
  return response.json()
}

function upload(file) {
  const body = new FormData()
  body.append('file', file)
  return { method: 'POST', body }
}

export function health() {
  return request('/health')
}

export function predict(file) {
  return request('/predict', upload(file))
}

// Los 16 frames que extrajo el pipeline real para un video ya analizado: el
// backend los guarda por `video_id` (el que devolvió /predict).
export function preprocess(videoId) {
  return request(`/preprocessing_demo?video_id=${encodeURIComponent(videoId)}`)
}

// Acepta el formato plano o el `metrics_<modelo>.json` que escribe train.py
// ({ model, final: {...} }), para no depender de como lo exponga la API.
export async function getMetrics() {
  const { final, ...rest } = await request('/metrics')
  return { ...rest, ...(final ?? {}) }
}
