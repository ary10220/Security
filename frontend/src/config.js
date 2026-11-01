// Configuracion del frontend. Los valores marcados PROVISIONAL son decisiones
// que el plan deja abiertas al equipo: se cambian aqui, no en los componentes.

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// Con VITE_USE_MOCK=true (o sin backend) la app usa datos simulados de
// src/lib/mock.js. Nunca se muestran como resultados reales: la interfaz los
// marca como "simulado".
export const USE_MOCK = (import.meta.env.VITE_USE_MOCK ?? 'false') === 'true'

export const MAX_UPLOAD_MB = 100 // PROVISIONAL: debe coincidir con la validacion del backend
export const ACCEPTED_TYPES = ['video/mp4']

// PROVISIONAL: umbral inicial del slider si la API no manda `default_threshold`.
export const DEFAULT_THRESHOLD = 0.5

// PROVISIONAL: el plan pide "nivel de alerta" pero no fija los cortes.
// `min` es el porcentaje de segmentos sospechosos desde el cual aplica el nivel.
export const ALERT_LEVELS = [
  { id: 'none', label: 'Sin alerta', min: 0 },
  { id: 'low', label: 'Alerta baja', min: 10 },
  { id: 'medium', label: 'Alerta media', min: 30 },
  { id: 'high', label: 'Alerta alta', min: 60 },
]
