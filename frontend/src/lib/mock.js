// Datos SIMULADOS para desarrollar el frontend sin backend. Nada de esto es un
// resultado del modelo ni una metrica medida del proyecto; la interfaz lo
// rotula como "simulado".
import { DEFAULT_THRESHOLD } from '../config.js'
import { extractFramesLocal, readDuration } from './frames.js'

const SEGMENT_SECONDS = 2 // PROVISIONAL: el equipo no ha confirmado 2 s o 3 s

// Generador pseudoaleatorio determinista (mulberry32) para que el mismo video
// de siempre el mismo timeline simulado.
function rng(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hash(text) {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619)
  return h >>> 0
}

export async function mockPredict(file) {
  const duration = await readDuration(file)
  const rand = rng(hash(file.name + file.size))
  const count = Math.max(1, Math.ceil(duration / SEGMENT_SECONDS))
  // Un pico gaussiano de "sospecha" en un punto del video + ruido.
  const center = rand() * count
  const width = 1 + rand() * 2
  const segments = Array.from({ length: count }, (_, i) => {
    const bump = Math.exp(-((i - center) ** 2) / (2 * width ** 2))
    const score = Math.min(1, Math.max(0, 0.1 + 0.8 * bump * rand() ** 0.3 + 0.1 * rand()))
    return {
      start: i * SEGMENT_SECONDS,
      end: Math.min((i + 1) * SEGMENT_SECONDS, duration),
      score: Number(score.toFixed(3)),
    }
  })
  return {
    source: 'mock',
    video: { name: file.name, duration_s: duration },
    segment_seconds: SEGMENT_SECONDS,
    model: 'simulado',
    default_threshold: DEFAULT_THRESHOLD,
    segments,
    clip_url: null,
  }
}

export async function mockPreprocess(file) {
  return extractFramesLocal(file)
}

export async function mockMetrics() {
  const recall = Array.from({ length: 41 }, (_, i) => 1 - i / 40)
  const precision = recall.map((r) => Math.min(1, 0.95 - 0.35 * r ** 3 + 0.05))
  const thresholds = recall.map((_, i) => Number((0.05 + (i / 40) * 0.9).toFixed(3)))
  return {
    source: 'mock',
    model: 'simulado',
    class_names: ['NonFight', 'Fight'],
    accuracy: 0.5,
    f1_macro: 0.5,
    f1_per_class: { NonFight: 0.5, Fight: 0.5 },
    precision_per_class: { NonFight: 0.5, Fight: 0.5 },
    recall_per_class: { NonFight: 0.5, Fight: 0.5 },
    confusion_matrix: [
      [50, 50],
      [50, 50],
    ],
    pr_curve: {
      precision,
      recall,
      thresholds,
      average_precision: 0.5,
      best_f1_threshold: 0.5,
      best_f1: 0.5,
    },
    inference_ms: null,
  }
}
