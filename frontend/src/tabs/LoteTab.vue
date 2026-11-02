<script setup>
import { computed, ref, watch } from 'vue'
import ModePicker from '../components/ModePicker.vue'
import Timeline from '../components/Timeline.vue'
import { MAX_UPLOAD_MB } from '../config.js'
import { useApi } from '../composables/useApi.js'
import { useHistorial } from '../composables/useHistorial.js'

const props = defineProps({
  modelReady: { type: Boolean, required: true },
  modes: { type: Array, required: true },
  mode: { type: String, required: true },
  modeDetail: { type: String, default: '' },
})
const emit = defineEmits(['update:mode'])
const api = useApi()
const { addPrediction } = useHistorial()
const files = ref([])
const rows = ref([])
const loading = ref(false)
const error = ref('')
const detail = ref(null)
const selectedCount = computed(() => files.value.length)

watch(() => props.mode, () => {
  // Se borran resultados de un modo al seleccionar otro.
  rows.value = []
  detail.value = null
  error.value = ''
})

function choose(event) {
  const selected = Array.from(event.target.files ?? [])
  error.value = ''
  rows.value = []
  if (selected.length > 20) {
    error.value = 'Selecciona como máximo 20 videos por lote.'
    return
  }
  if (selected.some((file) => !file.name.toLowerCase().endsWith('.mp4'))) {
    error.value = 'Todos los archivos deben ser .mp4.'
    return
  }
  if (selected.some((file) => file.size > MAX_UPLOAD_MB * 1024 * 1024)) {
    error.value = `Cada archivo debe pesar como máximo ${MAX_UPLOAD_MB} MB.`
    return
  }
  files.value = selected
}

async function analyzeBatch() {
  if (!files.value.length) return
  loading.value = true
  error.value = ''
  try {
    const requestedMode = props.mode
    const response = await api.predictBatch(files.value, requestedMode)
    if (requestedMode !== props.mode) return
    rows.value = response
    for (const row of rows.value) {
      if (row.result) addPrediction(row.filename, row.result)
    }
  } catch (cause) {
    error.value = cause.message
  } finally {
    loading.value = false
  }
}

function suspiciousProbability(row) {
  return Math.max(0, ...(row.result?.segments ?? []).map((segment) => segment.score))
}
</script>

<template>
  <section class="tab-content">
    <div class="section-heading"><div><p class="eyebrow">Análisis por lote</p><h2>Varios videos</h2></div></div>
    <ModePicker :modes="modes" :model-value="mode" group="batch-mode" @update:model-value="emit('update:mode', $event)" />
    <p v-if="!modelReady" class="banner" role="status"><strong>Modelo no disponible.</strong> {{ modeDetail || 'No hay checkpoint y métricas de prueba para este escenario.' }}</p>
    <div class="card">
      <label for="batch-files" class="label">Seleccionar hasta 20 videos MP4</label>
      <input id="batch-files" type="file" accept=".mp4,video/mp4" multiple :disabled="loading || !modelReady" @change="choose" />
      <p class="muted">{{ selectedCount }} archivo(s) seleccionado(s)</p>
      <button type="button" class="button" :disabled="loading || !selectedCount || !modelReady" @click="analyzeBatch">Analizar lote</button>
    </div>
    <div v-if="loading" class="card loading-line" role="status"><span class="spinner" /> Procesando lote en la API…</div>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <div v-if="rows.length" class="card table-wrap">
      <table>
        <thead><tr><th>Video</th><th>Prioridad</th><th>Puntuación máxima</th><th>Detalle</th></tr></thead>
        <tbody>
          <tr v-for="(row, index) in rows" :key="index">
            <td>{{ row.filename }}</td>
            <td v-if="row.result"><span class="badge" :class="row.result.verdict">
              {{ row.result.verdict === 'suspicious' ? 'REVISAR TRAMO' : 'SIN TRAMO PRIORITARIO' }}</span></td>
            <td v-else colspan="3" class="error">{{ row.error }}</td>
            <td v-if="row.result">{{ (suspiciousProbability(row) * 100).toFixed(1) }} %</td>
            <td v-if="row.result"><button type="button" @click="detail = row">Ver detalle</button></td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-if="detail" class="modal-backdrop" @click.self="detail = null">
      <div class="modal-panel" role="dialog" aria-modal="true" :aria-label="`Detalle de ${detail.filename}`">
        <div class="modal-header"><h2>{{ detail.filename }}</h2><button type="button" @click="detail = null">Cerrar</button></div>
        <p><span class="badge" :class="detail.result.verdict">{{ detail.result.verdict === 'suspicious' ? 'REVISAR TRAMO' : 'SIN TRAMO PRIORITARIO' }}</span>
          · Confianza {{ (detail.result.confidence * 100).toFixed(1) }} %</p>
        <p v-if="detail.result.verdict === 'suspicious'">Motivo: {{ detail.result.reason }}</p>
        <Timeline :segments="detail.result.segments" :threshold="detail.result.threshold"
          :duration="detail.result.video_duration" />
      </div>
    </div>
  </section>
</template>
