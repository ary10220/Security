import { describe, expect, it } from 'vitest'
import { MIN_TEST_CLIPS, isInconclusive } from './metrics.js'

describe('isInconclusive', () => {
  it('marca como no concluyente un test con pocos clips (exterior: 13)', () => {
    expect(isInconclusive({ n_test: 13 })).toBe(true)
  })

  it('acepta los tests con clips suficientes (interior: 62, peleas: 400)', () => {
    expect(isInconclusive({ n_test: 62 })).toBe(false)
    expect(isInconclusive({ n_test: 400 })).toBe(false)
  })

  it('el umbral es el mínimo: justo en el límite ya cuenta como concluyente', () => {
    expect(isInconclusive({ n_test: MIN_TEST_CLIPS })).toBe(false)
    expect(isInconclusive({ n_test: MIN_TEST_CLIPS - 1 })).toBe(true)
  })

  it('sin n_test no se oculta nada: no se puede afirmar que sea inconcluso', () => {
    expect(isInconclusive({})).toBe(false)
    expect(isInconclusive(undefined)).toBe(false)
  })
})
