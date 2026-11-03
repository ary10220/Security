// Mínimo de clips de test para publicar porcentajes. Con menos, el intervalo
// de confianza de la exactitud es tan ancho (exterior: 13 clips de 4 videos,
// 42–87 %) que una cifra sería ruido y engañaría más que informar.
export const MIN_TEST_CLIPS = 30

// Indica si la evaluación de un modelo no permite concluir nada. Se decide por
// el tamaño del test y no por el nombre del modo: si exterior se reentrena con
// más datos, la pestaña vuelve a mostrar sus cifras sin tocar el código.
export function isInconclusive(report) {
  const clips = report?.n_test
  return typeof clips === 'number' && clips < MIN_TEST_CLIPS
}
