// Logica pura del analisis: se recalcula en el navegador al mover el umbral,
// sin volver a llamar al backend. Cada segmento es {start, end, score} con
// `score` = probabilidad de "sospechoso" en [0, 1].
import { ALERT_LEVELS } from '../config.js'

export function flagSegments(segments, threshold) {
  return segments.map((s) => ({ ...s, suspicious: s.score >= threshold }))
}

export function alertLevel(percent, levels = ALERT_LEVELS) {
  let current = levels[0]
  for (const level of levels) if (percent >= level.min) current = level
  return current
}

export function summarize(segments, threshold, levels = ALERT_LEVELS) {
  const total = segments.length
  const flagged = segments.filter((s) => s.score >= threshold).length
  const percent = total === 0 ? 0 : (flagged / total) * 100
  return { total, flagged, percent, level: alertLevel(percent, levels) }
}

// Segmento con mayor probabilidad: es el "momento mas critico" del clip.
export function criticalSegment(segments) {
  if (segments.length === 0) return null
  return segments.reduce((best, s) => (s.score > best.score ? s : best))
}

// Une segmentos sospechosos consecutivos en tramos [{start, end, peak}].
export function suspiciousRuns(segments, threshold) {
  const runs = []
  for (const s of segments) {
    if (s.score < threshold) continue
    const last = runs[runs.length - 1]
    if (last && Math.abs(s.start - last.end) < 1e-6) {
      last.end = s.end
      last.peak = Math.max(last.peak, s.score)
    } else {
      runs.push({ start: s.start, end: s.end, peak: s.score })
    }
  }
  return runs
}

export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds))
  const m = Math.floor(total / 60)
  const s = String(total % 60).padStart(2, '0')
  return `${m}:${s}`
}
