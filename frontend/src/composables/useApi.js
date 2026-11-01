import { API_URL } from '../config.js'

async function request(path, options) {
  let response
  try {
    response = await fetch(`${API_URL}${path}`, options)
  } catch {
    throw new Error('No se pudo conectar con la API. Inicia el backend en localhost:8000.')
  }
  if (!response.ok) {
    let detail
    try {
      detail = (await response.json()).detail
    } catch {
      detail = null
    }
    throw new Error(typeof detail === 'string' ? detail : `La API respondió ${response.status}.`)
  }
  return response.json()
}

export function useApi() {
  // Se envían FormData sin Content-Type manual para que el navegador agregue el boundary.
  return {
    health: () => request('/health'),
    metrics: (model) => request(`/metrics?model=${encodeURIComponent(model)}`),
    predict: (file) => {
      const body = new FormData()
      body.append('file', file)
      return request('/predict', { method: 'POST', body })
    },
    predictBatch: (files) => {
      const body = new FormData()
      for (const file of files) body.append('files', file)
      return request('/predict/batch', { method: 'POST', body })
    },
    preprocessingDemo: (videoId) =>
      request(`/preprocessing_demo?video_id=${encodeURIComponent(videoId)}`),
  }
}
