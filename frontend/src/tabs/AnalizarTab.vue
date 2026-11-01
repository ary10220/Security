<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import UploadDropzone from '../components/UploadDropzone.vue'
import Timeline from '../components/Timeline.vue'
import ThresholdSlider from '../components/ThresholdSlider.vue'
import PreprocessView from '../components/PreprocessView.vue'
import { deriveVerdict } from '../lib/analysis.js'
import { useApi } from '../composables/useApi.js'
import { useHistorial } from '../composables/useHistorial.js'

defineProps({ modelReady: { type: Boolean, required: true } })
const api = useApi()
const { addPrediction } = useHistorial()
const file = ref(null)
const videoUrl = ref('')
const video = ref(null)
const currentTime = ref(0)
const result = ref(null)
const threshold = ref(0.5)
const loading = ref(false)
const error = ref('')
const modalOpen = ref(false)
const frames = ref(null)
const framesLoading = ref(false)
const framesError = ref('')
const verdict = computed(() => result.value
  ? deriveVerdict(result.value.segments, threshold.value) : null)
const suspiciousProbability = computed(() => Math.max(
  0, ...(result.value?.segments ?? []).map((segment) => segment.score),
))

onBeforeUnmount(() => { if (videoUrl.value) URL.revokeObjectURL(videoUrl.value) })

async function analyze(selected) {
  // La vista previa usa el archivo local; la inferencia se hace en la API.
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  file.value = selected
  videoUrl.value = URL.createObjectURL(selected)
  result.value = null
  frames.value = null
  error.value = ''
  loading.value = true
  try {
    result.value = await api.predict(selected)
    threshold.value = result.value.default_threshold ?? 0.5
    addPrediction(selected.name, result.value)
  } catch (cause) {
    error.value = cause.message
  } finally {
    loading.value = false
  }
}

async function showFrames() {
  if (!result.value) return
  modalOpen.value = true
  if (frames.value) return
  framesLoading.value = true
  framesError.value = ''
  try {
    frames.value = await api.preprocessingDemo(result.value.video_id)
  } catch (cause) {
    framesError.value = cause.message
  } finally {
    framesLoading.value = false
  }
}

function seek(time) {
  if (video.value) video.value.currentTime = time
}
</script>

<template>
  <section class="tab-content">
    <div class="section-heading">
      <div><p class="eyebrow">Análisis individual</p><h2>Revisar un video</h2></div>
      <p class="muted">Archivo MP4 · máximo 100 MB · hasta 2 minutos</p>
    </div>
    <p v-if="!modelReady" class="banner" role="status">La inferencia requiere que la API tenga un checkpoint entrenado cargado.</p>
    <UploadDropzone :disabled="loading || !modelReady" @select="analyze" />
    <div v-if="loading" class="card loading-line" role="status"><span class="spinner" /> Analizando {{ file?.name }}…</div>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="result">
      <div class="analysis-grid">
        <video ref="video" :src="videoUrl" controls @timeupdate="currentTime = $event.target.currentTime" />
        <section class="card result-card" :class="verdict.verdict">
          <p class="eyebrow">Veredicto con umbral {{ threshold.toFixed(2) }}</p>
          <strong class="verdict-word">{{ verdict.verdict === 'suspicious' ? 'SOSPECHOSO' : 'NORMAL' }}</strong>
          <p class="muted">Probabilidad máxima de Sospechoso: {{ (suspiciousProbability * 100).toFixed(1) }} %</p>
          <div class="probability-track"><span :style="{ width: (suspiciousProbability * 100) + '%' }" /></div>
          <p class="muted">Confianza del veredicto: {{ (verdict.confidence * 100).toFixed(1) }} %</p>
        </section>
      </div>
      <ThresholdSlider v-model="threshold" />
      <Timeline :segments="result.segments" :threshold="threshold"
        :duration="result.video_duration" :current-time="currentTime" @seek="seek" />
      <button type="button" class="button" @click="showFrames">Ver preprocesamiento</button>
    </template>
    <p v-else-if="!loading && !error" class="muted">Sube un video para ver el resultado y la línea de tiempo.</p>

    <div v-if="modalOpen" class="modal-backdrop" @click.self="modalOpen = false">
      <div class="modal-panel" role="dialog" aria-modal="true" aria-label="Preprocesamiento">
        <div class="modal-header"><h2>16 frames extraídos</h2><button type="button" @click="modalOpen = false">Cerrar</button></div>
        <p v-if="framesLoading" class="muted">Cargando frames…</p>
        <p v-if="framesError" class="error" role="alert">{{ framesError }}</p>
        <PreprocessView v-if="frames" :data="frames" />
      </div>
    </div>
  </section>
</template>
