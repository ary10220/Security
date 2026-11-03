<script setup>
import { computed, onMounted, ref } from 'vue'
import OpticalLens from './OpticalLens.vue'
import { useApi } from './composables/useApi.js'

const models = [
  { id: 'mobilenet_lstm', name: 'Peleas · MobileNet + BiLSTM', detail: 'Fight / NonFight · RWF-2000. Esta copia local carga la corrida CPU de 100 clips de test.' },
  { id: 'mobilenet_baseline', name: 'Peleas · MobileNet un frame', detail: 'Fight / NonFight · RWF-2000. Referencia de un fotograma.' },
  { id: 'resnet_lstm', name: 'Peleas · ResNet + BiLSTM', detail: 'Fight / NonFight · RWF-2000 completo. Corrida GPU: 400 clips de test. Este botón compara métricas; Analizar usa MobileNet.' },
  { id: 'interior', name: 'Riesgo observable en interiores · ASHEN', detail: 'Normal / Riesgo · Robbery y Assault agrupados. Test propio: 62 clips de 18 videos fuente.' },
  { id: 'exterior', name: 'Riesgo observable en exteriores · ASHEN', detail: 'Normal / Riesgo · se entrenó, pero no hubo tiempo de ampliar sus datos ni validarlo más. Test: 13 clips de 4 videos fuente.' },
]
const stages = [
  { name: 'Encuadra', title: 'Del video a 16 fotogramas', text: 'De cada tramo se toman 16 imágenes a intervalos regulares. Puedes verlas junto al resultado para saber qué recibió el modelo.' },
  { name: 'Señala', title: 'Del video a una señal', text: 'Según el modo elegido, el modelo puntúa similitud con clips de peleas o con eventos de riesgo observables. No determina si ocurrió un delito.' },
  { name: 'Revisa', title: 'De la señal a la decisión humana', text: 'Un operador revisa el tramo y su contexto. Solo esa persona decide si corresponde contactar a la policía.' },
]
const stage = ref(0)
const selected = ref(models[0])
const reports = ref({})
const modeStatuses = ref([])
const loading = ref(true)
const report = computed(() => reports.value[selected.value.id])
const visibleModels = computed(() => loading.value ? [] : models.filter(model => reports.value[model.id]?.model_size_mb != null))
const riskMode = computed(() => selected.value.id === 'interior' || selected.value.id === 'exterior')
const inconclusive = computed(() => selected.value.id === 'exterior')
const sourceCount = computed(() => selected.value.id === 'interior' ? 18 : selected.value.id === 'exterior' ? 4 : null)
const percent = n => n == null ? '—' : new Intl.NumberFormat('es', { style: 'percent', maximumFractionDigits: 1 }).format(n)
onMounted(async () => {
  const api = useApi()
  try { modeStatuses.value = await api.modes() } catch { modeStatuses.value = [] }
  const results = await Promise.allSettled(models.map(model => api.metrics(model.id)))
  results.forEach((result, i) => {
    if (result.status === 'fulfilled') reports.value[models[i].id] = result.value
  })
  loading.value = false
})
</script>

<template>
  <section class="hero">
    <div class="hero-inner rail">
      <div class="hero-copy">
        <span class="kicker"><i></i> SEGURIDAD PARA COMERCIOS Y EMPRESAS</span>
        <h1 class="hero-headline">Vigía ayuda a encontrar momentos<br><em>que requieren revisión en cámaras de seguridad.</em></h1>
        <p>Cámara o video → señal de riesgo → revisión por una persona → decisión de contactar a la policía. Vigía ayuda a priorizar tramos; no predice asaltos ni identifica armas reales.</p>
        <div class="hero-actions"><a class="v-cta" href="#analizar">Probar con mi video <b aria-hidden="true">↗</b></a><a class="text-link" href="#metodo">Cómo funciona <span>↓</span></a></div>
      </div>
      <OpticalLens />
    </div>
    <div class="hero-foot rail"><span>VIGÍA <i>/</i> VISIÓN POR COMPUTADORA</span><span>UAGRM <i>/</i> PROTOTIPO ACADÉMICO</span></div>
  </section>
  <section id="metodo" class="method rail">
    <div><span class="folio">01 / EL MÉTODO</span><h2>Una segunda<br><em>mirada.</em></h2><p>Desde el mostrador de una tienda hasta el acceso de una empresa, el recorrido empieza con un video de la cámara de seguridad.</p></div>
    <div class="method-list"><div v-for="(step, i) in stages" :key="step.name" class="method-row" :class="{ chosen: stage === i }">
      <button :aria-expanded="stage === i" :aria-controls="'paso-'+i" @click="stage = stage === i ? -1 : i"><small>0{{ i+1 }}</small><strong>{{ step.name }}</strong><span>{{ stage === i ? '−' : '+' }}</span></button>
      <div v-show="stage === i" :id="'paso-'+i" class="method-body"><b>{{ step.title }}</b><p>{{ step.text }}</p></div>
    </div></div>
  </section>
  <section id="modelos" class="evidence">
    <div class="rail"><span class="folio">02 / EVIDENCIA</span><div class="evidence-title"><h2>El modelo, <em>a prueba.</em></h2><a href="#metricas" class="text-link">Explorar métricas ↗</a></div>
      <p>Elige un modelo para ver su evaluación propia. Peleas usa RWF-2000; los modos de riesgo usan ASHEN con etiquetas Normal/Riesgo. Ningún porcentaje de peleas mide robos.</p>
      <p class="muted">Ajustes locales de presentación; no forman parte de los PR entregados. Prototipo académico: ASHEN tiene licencia no comercial.</p>
      <p v-if="modeStatuses.length" class="mode-status-line">Modos: <span v-for="item in modeStatuses" :key="item.id">{{ item.label }} · {{ item.available ? 'disponible' : 'Modelo no disponible' }}. </span></p>
      <div class="model-switch" aria-label="Arquitecturas"><button v-for="model in visibleModels" :key="model.id" :aria-pressed="selected.id === model.id" @click="selected = model">{{ model.name }}</button></div>
      <div class="report-strip" aria-live="polite"><div class="report-note"><b>{{ selected.name }}</b><p>{{ selected.detail }}</p><p v-if="report">{{ report.n_test }} clips de test<span v-if="sourceCount"> · {{ sourceCount }} videos fuente</span><span v-else> · fuentes no registradas en este reporte</span></p><p v-if="inconclusive"><em>No concluyente: resultado preliminar con solo 4 videos fuente.</em></p></div>
        <template v-if="report"><div><strong>{{ percent(report.final.accuracy) }}</strong><small>Exactitud observada en test{{ inconclusive ? ' · no concluyente' : '' }}</small></div><div><strong>{{ percent(report.final.recall_per_class[riskMode ? 'Riesgo' : 'Fight']) }}</strong><small>Recall {{ riskMode ? 'Riesgo' : 'Fight' }}</small></div><div><strong>{{ percent(report.final.f1_macro) }}</strong><small>F1 macro</small></div></template>
        <p v-else class="report-missing">{{ loading ? 'Consultando la evaluación…' : 'Modelo no disponible: falta su evaluación o checkpoint local.' }}</p>
      </div>
      <div class="evidence-links"><a href="#datos">Ver datos y particiones ↗</a><a href="#lote">Analizar varios videos ↗</a></div>
    </div>
  </section>
</template>
