<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import MetricsPanel from './components/MetricsPanel.vue'
import PreprocessView from './components/PreprocessView.vue'
import ThresholdSlider from './components/ThresholdSlider.vue'
import Timeline from './components/Timeline.vue'
import UploadDropzone from './components/UploadDropzone.vue'
import VerdictCard from './components/VerdictCard.vue'
import { DEFAULT_THRESHOLD, USE_MOCK } from './config.js'
import * as api from './lib/api.js'

const tabs = [
  { id: 'result', label: 'Resultado' },
  { id: 'metrics', label: 'Métricas del modelo' },
  { id: 'preprocess', label: 'Preprocesamiento' },
]

const tab = ref('result')
const file = ref(null)
const videoUrl = ref(null)
const video = ref(null)
const currentTime = ref(0)
const loading = ref(false)
const error = ref('')
const result = ref(null)
const preprocessed = ref(null)
const metrics = ref(null)
const metricsError = ref('')
const threshold = ref(DEFAULT_THRESHOLD)

const duration = computed(() => result.value?.video.duration_s ?? 0)

onMounted(async () => {
  try {
    metrics.value = await api.getMetrics()
  } catch (e) {
    metricsError.value = e.message
  }
})
onBeforeUnmount(() => videoUrl.value && URL.revokeObjectURL(videoUrl.value))

async function analyze(selected) {
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  file.value = selected
  videoUrl.value = URL.createObjectURL(selected)
  result.value = null
  preprocessed.value = null
  error.value = ''
  loading.value = true
  try {
    result.value = await api.predict(selected)
    threshold.value = result.value.default_threshold ?? DEFAULT_THRESHOLD
    // La vista de preprocesamiento es secundaria: si falla no tumba el resultado.
    api.preprocess(selected).then(
      (data) => (preprocessed.value = data),
      () => (preprocessed.value = null),
    )
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function seek(time) {
  if (!video.value) return
  video.value.currentTime = time
  video.value.play()
}
</script>

<template>
  <header class="top">
    <h1>Detección de comportamiento sospechoso</h1>
    <p class="muted">Ayuda para priorizar la revisión humana de video de cámaras de seguridad.</p>
    <p v-if="USE_MOCK" class="banner" role="status">
      Modo simulado: el análisis no usa el modelo real (backend no conectado).
    </p>
  </header>

  <main>
    <UploadDropzone :disabled="loading" @select="analyze" />

    <nav class="tabs" role="tablist">
      <button
        v-for="t in tabs"
        :key="t.id"
        type="button"
        role="tab"
        :aria-selected="tab === t.id"
        :class="{ active: tab === t.id }"
        @click="tab = t.id"
      >
        {{ t.label }}
      </button>
    </nav>

    <p v-if="loading" class="muted" role="status">Analizando {{ file?.name }}…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>

    <template v-if="tab === 'result'">
      <p v-if="!result && !loading && !error" class="muted">Sube un video para ver el análisis.</p>
      <template v-if="result">
        <video ref="video" :src="videoUrl" controls @timeupdate="currentTime = $event.target.currentTime" />
        <VerdictCard :segments="result.segments" :threshold="threshold" :clip-url="result.clip_url" />
        <ThresholdSlider
          v-model="threshold"
          :best-threshold="metrics?.source === 'api' ? (metrics.pr_curve?.best_f1_threshold ?? null) : null"
        />
        <Timeline
          :segments="result.segments"
          :threshold="threshold"
          :duration="duration"
          :current-time="currentTime"
          @seek="seek"
        />
      </template>
    </template>

    <template v-else-if="tab === 'metrics'">
      <MetricsPanel v-if="metrics" :metrics="metrics" :threshold="threshold" />
      <p v-else-if="metricsError" class="error" role="alert">{{ metricsError }}</p>
      <p v-else class="muted">Cargando métricas…</p>
    </template>

    <template v-else>
      <PreprocessView v-if="preprocessed" :data="preprocessed" />
      <p v-else class="muted">Sube un video para ver los frames que recibe el modelo.</p>
    </template>
  </main>
</template>
