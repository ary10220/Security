import { ref } from 'vue'

const KEY = 'security-predictions-v1'
const items = ref([])

try {
  const saved = JSON.parse(localStorage.getItem(KEY) || '[]')
  items.value = Array.isArray(saved) ? saved.slice(0, 20) : []
} catch {
  items.value = []
}

export function useHistorial() {
  function addPrediction(filename, result) {
    // El historial conserva solo datos breves; no almacena el video ni frames base64.
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      date: new Date().toISOString(),
      filename,
      verdict: result.verdict,
      confidence: result.confidence,
    }
    items.value = [entry, ...items.value].slice(0, 20)
    try {
      localStorage.setItem(KEY, JSON.stringify(items.value))
    } catch {
      // El modo privado puede impedir escritura; la lista sigue en memoria.
    }
  }

  return { items, addPrediction }
}
