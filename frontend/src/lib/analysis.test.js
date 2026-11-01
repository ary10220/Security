import { describe, expect, it } from 'vitest'
import {
  alertLevel,
  criticalSegment,
  deriveVerdict,
  flagSegments,
  formatTime,
  suspiciousRuns,
  summarize,
} from './analysis.js'

const seg = (start, score) => ({ start, end: start + 2, score })
const segments = [seg(0, 0.1), seg(2, 0.8), seg(4, 0.9), seg(6, 0.2), seg(8, 0.7)]

describe('analysis', () => {
  it('marca como sospechoso lo que alcanza el umbral (>=)', () => {
    const flags = flagSegments(segments, 0.7).map((s) => s.suspicious)
    expect(flags).toEqual([false, true, true, false, true])
  })

  it('el porcentaje cambia al mover el umbral', () => {
    expect(summarize(segments, 0.5).percent).toBe(60)
    expect(summarize(segments, 0.85).percent).toBe(20)
    expect(summarize(segments, 0.95).flagged).toBe(0)
  })

  it('asigna el nivel de alerta segun el porcentaje', () => {
    expect(alertLevel(0).id).toBe('none')
    expect(alertLevel(10).id).toBe('low')
    expect(alertLevel(59.9).id).toBe('medium')
    expect(alertLevel(100).id).toBe('high')
  })

  it('sin segmentos no divide por cero', () => {
    expect(summarize([], 0.5)).toMatchObject({ total: 0, flagged: 0, percent: 0 })
    expect(criticalSegment([])).toBeNull()
  })

  it('el momento critico es el de mayor probabilidad', () => {
    expect(criticalSegment(segments).start).toBe(4)
  })

  it('une segmentos consecutivos en tramos', () => {
    expect(suspiciousRuns(segments, 0.7)).toEqual([
      { start: 2, end: 6, peak: 0.9 },
      { start: 8, end: 10, peak: 0.7 },
    ])
  })

  it('formatea el tiempo como m:ss', () => {
    expect(formatTime(0)).toBe('0:00')
    expect(formatTime(75.9)).toBe('1:15')
  })

  it('recalcula el veredicto con la ventana temporal de la API', () => {
    expect(deriveVerdict(segments, 0.7).verdict).toBe('suspicious')
    expect(deriveVerdict(segments, 0.85).verdict).toBe('normal')
    expect(deriveVerdict([seg(0, 0.8)], 0.5).verdict).toBe('suspicious')
  })
})
