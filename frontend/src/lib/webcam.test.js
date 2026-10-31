import { describe, expect, it } from 'vitest'
import { pickRecordingType, recordingFile } from './webcam.js'

describe('pickRecordingType', () => {
  it('elige el primer formato que el navegador soporta', () => {
    const soportados = new Set(['video/webm', 'video/mp4'])
    const type = pickRecordingType((t) => soportados.has(t), ['video/mp4', 'video/webm'])
    expect(type).toBe('video/mp4')
  })

  it('devuelve null si el navegador no soporta ninguno', () => {
    expect(pickRecordingType(() => false, ['video/mp4'])).toBeNull()
  })

  it('por defecto solo ofrece formatos que la API acepta (mp4)', () => {
    expect(pickRecordingType(() => true)).toBe('video/mp4')
  })
})

describe('recordingFile', () => {
  const ahora = new Date('2026-10-31T16:30:05Z')

  it('nombra el archivo con la fecha y la extensión del formato real', () => {
    const file = recordingFile([new Blob(['x'])], 'video/mp4', ahora)
    expect(file.name).toBe('webcam-2026-10-31-16-30-05.mp4')
    expect(file.type).toBe('video/mp4')
  })

  it('quita los codecs del tipo y no cambia la extensión de un webm', () => {
    const file = recordingFile([new Blob(['x'])], 'video/webm;codecs=vp9', ahora)
    expect(file.name.endsWith('.webm')).toBe(true)
    expect(file.type).toBe('video/webm')
  })
})
