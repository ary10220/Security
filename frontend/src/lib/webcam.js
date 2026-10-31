// Lógica de la grabación con webcam, separada del componente para poder
// probarla sin navegador ni cámara.
import { RECORDING_TYPES } from '../config.js'

// Devuelve el primer formato que el navegador sabe grabar y que la API acepta,
// o null si no hay ninguno. `isTypeSupported` es MediaRecorder.isTypeSupported.
// Se pasa como parámetro para no depender de MediaRecorder en los tests.
export function pickRecordingType(isTypeSupported, candidates = RECORDING_TYPES) {
  return candidates.find((type) => isTypeSupported(type)) ?? null
}

// Arma el archivo que se envía a /predict con los trozos que entregó el
// MediaRecorder. La extensión sale del formato real de la grabación: no se
// renombra un .webm a .mp4 para engañar la validación del backend.
export function recordingFile(chunks, type, now = new Date()) {
  const extension = type.split('/')[1].split(';')[0]
  const stamp = now.toISOString().slice(0, 19).replace(/[T:]/g, '-')
  return new File(chunks, `webcam-${stamp}.${extension}`, { type: type.split(';')[0] })
}
