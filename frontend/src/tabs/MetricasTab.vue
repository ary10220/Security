<script setup>
import { computed, onMounted, ref } from 'vue'
import PrCurve from '../components/PrCurve.vue'
import { useApi } from '../composables/useApi.js'

const api = useApi()
const modelNames = ['mobilenet_lstm', 'mobilenet_baseline', 'resnet_lstm', 'interior', 'exterior']
const reports = ref({})
const selected = ref('mobilenet_lstm')
const loading = ref(true)
const error = ref('')
const available = computed(() => modelNames.filter((name) => reports.value[name]?.model_size_mb != null))
const report = computed(() => reports.value[selected.value])
const final = computed(() => report.value?.final)
const riskMode = computed(() => report.value?.mode === 'interior' || report.value?.mode === 'exterior')
const inconclusive = computed(() => report.value?.mode === 'exterior')
const sourceCount = computed(() => report.value?.mode === 'interior' ? 18 : report.value?.mode === 'exterior' ? 4 : null)
const classLabels = { NonFight: 'Sin pelea', Fight: 'Pelea', Normal: 'Normal', Riesgo: 'Riesgo observable' }
const displayName = {
  mobilenet_lstm: 'Peleas · MobileNet + BiLSTM',
  mobilenet_baseline: 'Peleas · MobileNet un frame',
  resnet_lstm: 'Peleas · ResNet + BiLSTM',
  interior: 'Riesgo observable en interiores · ASHEN',
  exterior: 'Riesgo observable en exteriores · ASHEN',
}
const pct = (value) => value == null ? 'Sin medir' : `${(value * 100).toFixed(1)} %`

onMounted(async () => {
  const results = await Promise.allSettled(modelNames.map((name) => api.metrics(name)))
  const next = {}
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') next[modelNames[index]] = result.value
  })
  reports.value = next
  const usable = modelNames.filter((name) => next[name]?.model_size_mb != null)
  if (!usable.includes(selected.value)) selected.value = usable[0] ?? ''
  if (!usable.length) error.value = 'Modelo no disponible: faltan checkpoints o métricas en la API.'
  loading.value = false
})
</script>

<template>
  <section class="tab-content">
    <div class="section-heading">
      <div><p class="eyebrow">Evaluación del modelo</p><h2>Métricas de test</h2></div>
      <label v-if="available.length" class="model-picker">Modelo
        <select v-model="selected"><option v-for="name in available" :key="name" :value="name">{{ displayName[name] }}</option></select>
      </label>
    </div>
    <p v-if="loading" class="muted">Cargando métricas…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <template v-if="report && final">
      <p class="muted">{{ riskMode ? 'Normal / Riesgo · ASHEN' : 'Fight / NonFight · RWF-2000' }} · {{ report.n_test }} clips de test<span v-if="sourceCount"> de {{ sourceCount }} videos fuente</span><span v-else> · fuentes no registradas en el JSON</span> · {{ report.evaluated_on }}.</p>
      <p v-if="riskMode" class="muted">Riesgo agrupa Robbery y Assault; el checkpoint no predice una clase Robbery ni reconoce armas. Los datos ASHEN son de uso no comercial.</p>
      <p v-else class="muted">Corrida {{ report.device?.toUpperCase() ?? 'sin dispositivo registrado' }} de {{ report.n_test }} clips de test. Las corridas CPU de 100 clips y GPU de 400 tienen protocolos distintos; no representan una mejora comparable. La comparación 8/16/32 frames corresponde a otros pesos.</p>
      <p v-if="!reports.interior?.model_size_mb || !reports.exterior?.model_size_mb" class="muted">Interiores: {{ reports.interior?.model_size_mb ? 'evaluado' : 'Modelo no disponible' }} · Exteriores: {{ reports.exterior?.model_size_mb ? 'evaluado' : 'Modelo no disponible' }}.</p>
      <p v-if="inconclusive" class="banner" role="status"><strong>No concluyente.</strong> Exterior sí se entrenó, pero no hubo tiempo de ampliar los datos ni validarlo más. Se muestran los resultados observados en 13 clips de solo 4 videos fuente; no estiman un rendimiento confiable.</p>
      <div class="metrics-summary">
        <div class="card stat"><span>Exactitud{{ inconclusive ? ' observada · no concluyente' : '' }}</span><strong>{{ pct(final.accuracy) }}</strong></div>
        <div class="card stat"><span>F1 macro</span><strong>{{ pct(final.f1_macro) }}</strong></div>
        <div class="card stat"><span>Recall {{ riskMode ? 'Riesgo' : 'Fight' }}</span><strong>{{ pct(final.recall_per_class[riskMode ? 'Riesgo' : 'Fight']) }}</strong></div>
        <div class="card stat"><span>AP</span><strong>{{ pct(report.pr_curve?.average_precision) }}</strong></div>
        <div class="card stat"><span>Tiempo / clip</span><strong>{{ report.inference_ms == null ? 'Sin medir' : report.inference_ms.toFixed(1) + ' ms' }}</strong></div>
        <div class="card stat"><span>Tamaño</span><strong>{{ report.model_size_mb == null ? 'Sin medir' : report.model_size_mb + ' MB' }}</strong></div>
        <div v-if="report.mode === 'interior' || report.mode === 'exterior'" class="card stat"><span>Falsas alertas en test</span><strong>{{ report.false_alerts ?? 'Sin medir' }}</strong></div>
      </div>
      <div class="metric-grid">
        <section class="card">
          <h3>Matriz de confusión</h3>
          <p class="muted">Filas: clase real · columnas: predicción</p>
          <div class="confusion-grid">
            <div></div><div>{{ riskMode ? 'Normal' : 'NonFight' }}</div><div>{{ riskMode ? 'Riesgo' : 'Fight' }}</div>
            <div>{{ riskMode ? 'Normal' : 'NonFight' }}</div>
            <div class="confusion-cell correct">{{ final.confusion_matrix[0][0] }}<small>TN</small></div>
            <div class="confusion-cell wrong">{{ final.confusion_matrix[0][1] }}<small>FP</small></div>
            <div>{{ riskMode ? 'Riesgo' : 'Fight' }}</div>
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
      <section v-if="report.test_errors?.length" class="card">
        <h3>Ejemplos de errores en test</h3>
        <ul><li v-for="item in report.test_errors.slice(0, 20)" :key="item.clip">{{ item.type === 'false_alert' ? 'Falsa alerta' : 'Evento omitido' }} · {{ item.clip }} · puntuación {{ pct(item.score) }}</li></ul>
      </section>
    </template>
  </section>
</template>
