// Configuracion del frontend. Los valores marcados PROVISIONAL son decisiones
// que el plan deja abiertas al equipo: se cambian aqui, no en los componentes.

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export const MAX_UPLOAD_MB = 100 // PROVISIONAL: debe coincidir con la validacion del backend
export const ACCEPTED_TYPES = ['video/mp4']

// El plan pide grabar 10 s con la webcam. La API solo acepta mp4/avi/mov, y los
// navegadores graban en mp4 o webm: se ofrece solo mp4 hasta que el backend
// acepte webm (pendiente de confirmar con quien hace api/).
export const WEBCAM_SECONDS = 10
export const RECORDING_TYPES = ['video/mp4']

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
