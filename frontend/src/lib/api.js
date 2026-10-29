// Cliente del backend FastAPI (endpoints del plan: /health, /metrics,
// /predict, /preprocess, /report). Los formatos de respuesta son una PROPUESTA
// del frontend (ver frontend/README.md) pendiente de confirmar con quien haga
// la API. Con USE_MOCK se usan datos simulados.
import { API_URL, USE_MOCK } from '../config.js'
import { mockMetrics, mockPredict, mockPreprocess } from './mock.js'

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

export async function health() {
  if (USE_MOCK) return { status: 'mock' }
  return request('/health')
}

export async function predict(file) {
  if (USE_MOCK) return mockPredict(file)
  return request('/predict', upload(file))
}

export async function preprocess(file) {
  if (USE_MOCK) return mockPreprocess(file)
  return request('/preprocess', upload(file))
}

// Acepta el formato plano o el `metrics_<modelo>.json` que escribe train.py
// ({ model, final: {...} }), para no depender de como lo exponga la API.
export async function getMetrics() {
  const raw = USE_MOCK ? await mockMetrics() : await request('/metrics')
  const { final, ...rest } = raw
  return { source: 'api', ...rest, ...(final ?? {}) }
}

export function reportUrl() {
  return `${API_URL}/report`
}
