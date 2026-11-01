<script setup>
import { computed, onMounted, ref } from 'vue'
import PrCurve from '../components/PrCurve.vue'
import { useApi } from '../composables/useApi.js'

const api = useApi()
const modelNames = ['mobilenet_lstm', 'mobilenet_baseline', 'resnet_lstm']
const reports = ref({})
const selected = ref('mobilenet_lstm')
const loading = ref(true)
const error = ref('')
const available = computed(() => modelNames.filter((name) => reports.value[name]))
const report = computed(() => reports.value[selected.value])
const final = computed(() => report.value?.final)
const classLabels = { NonFight: 'Normal', Fight: 'Sospechoso' }
const pct = (value) => value == null ? 'Sin medir' : `${(value * 100).toFixed(1)} %`

onMounted(async () => {
  const results = await Promise.allSettled(modelNames.map((name) => api.metrics(name)))
  const next = {}
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') next[modelNames[index]] = result.value
  })
  reports.value = next
  if (!next[selected.value]) selected.value = Object.keys(next)[0] ?? ''
  if (!Object.keys(next).length) error.value = 'No se pudieron cargar las métricas desde la API.'
  loading.value = false
})
</script>

<template>
  <section class="tab-content">
    <div class="section-heading">
      <div><p class="eyebrow">Evaluación del modelo</p><h2>Métricas de test</h2></div>
      <label v-if="available.length" class="model-picker">Modelo
        <select v-model="selected"><option v-for="name in available" :key="name" :value="name">{{ name }}</option></select>
      </label>
    </div>
    <p v-if="loading" class="muted">Cargando métricas…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="report && final">
      <p class="muted">Evaluación: {{ report.evaluated_on }}. ResNet LSTM se mostrará cuando exista su JSON de test.</p>
      <div class="metrics-summary">
        <div class="card stat"><span>Accuracy</span><strong>{{ pct(final.accuracy) }}</strong></div>
        <div class="card stat"><span>F1 macro</span><strong>{{ pct(final.f1_macro) }}</strong></div>
        <div class="card stat"><span>AP</span><strong>{{ pct(report.pr_curve?.average_precision) }}</strong></div>
        <div class="card stat"><span>Tiempo / clip</span><strong>{{ report.inference_ms == null ? 'Sin medir' : report.inference_ms.toFixed(1) + ' ms' }}</strong></div>
        <div class="card stat"><span>Tamaño</span><strong>{{ report.model_size_mb == null ? 'Sin medir' : report.model_size_mb + ' MB' }}</strong></div>
      </div>
      <div class="metric-grid">
        <section class="card">
          <h3>Matriz de confusión</h3>
          <p class="muted">Filas: clase real · columnas: predicción</p>
          <div class="confusion-grid">
            <div></div><div>Normal</div><div>Sospechoso</div>
            <div>Normal</div>
            <div class="confusion-cell correct">{{ final.confusion_matrix[0][0] }}<small>TN</small></div>
            <div class="confusion-cell wrong">{{ final.confusion_matrix[0][1] }}<small>FP</small></div>
            <div>Sospechoso</div>
            <div class="confusion-cell wrong">{{ final.confusion_matrix[1][0] }}<small>FN</small></div>
            <div class="confusion-cell correct">{{ final.confusion_matrix[1][1] }}<small>TP</small></div>
          </div>
        </section>
        <section class="card">
          <h3>Resultados por clase</h3>
          <div class="table-wrap"><table>
            <thead><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody><tr v-for="name in report.class_names" :key="name">
              <td>{{ classLabels[name] ?? name }}</td>
              <td>{{ pct(final.precision_per_class[name]) }}</td>
              <td>{{ pct(final.recall_per_class[name]) }}</td>
              <td>{{ pct(final.f1_per_class[name]) }}</td>
            </tr></tbody>
          </table></div>
        </section>
      </div>
      <div class="card" v-if="report.pr_curve">
        <h3>Curva precisión–recall</h3>
        <PrCurve :curve="report.pr_curve" :threshold="0.5" />
      </div>
    </template>
  </section>
</template>
