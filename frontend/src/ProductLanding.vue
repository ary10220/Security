<script setup>
import { computed, onMounted, ref } from 'vue'
import OpticalLens from './OpticalLens.vue'
import { useApi } from './composables/useApi.js'

const models = [
  { id: 'mobilenet_lstm', name: 'MobileNet + BiLSTM', detail: 'Lee el movimiento a lo largo de 16 fotogramas.' },
  { id: 'mobilenet_baseline', name: 'MobileNet · un frame', detail: 'Clasifica el fotograma central. Sirve como referencia para medir cuánto aporta el movimiento.' },
  { id: 'resnet_lstm', name: 'ResNet + BiLSTM', detail: 'Extrae rasgos con ResNet50 y compara su evolución en la secuencia.' },
]
const stages = [
  { name: 'Encuadra', title: 'Del video a 16 fotogramas', text: 'De cada tramo se toman 16 imágenes a intervalos regulares. Puedes verlas junto al resultado para saber qué recibió el modelo.' },
  { name: 'Contrasta', title: 'Del movimiento a una señal', text: 'La red combina los rasgos de cada imagen con su orden temporal y asigna una puntuación de pelea a cada tramo.' },
  { name: 'Revisa', title: 'Del aviso a la decisión', text: 'Abre el tramo señalado y observa qué sucede antes y después. Con ese contexto, la persona a cargo decide si corresponde avisar a la policía.' },
]
const stage = ref(0)
const selected = ref(models[0])
const reports = ref({})
const modeStatuses = ref([])
const loading = ref(true)
const report = computed(() => reports.value[selected.value.id])
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
        <h1>Tu negocio,<br><em>en foco.</em></h1>
        <p>Seguridad para todo tipo de local, cada minuto cuenta. Vigía señala tramos de la grabación, alerta para que tu equipo revise lo ocurrido y decida si debe llamar a la policía.</p>
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
      <p>Estas cifras corresponden únicamente a peleas y no peleas de RWF-2000. Los modos Interior / joyería y Exterior necesitarán clips propios etiquetados y una evaluación separada antes de ofrecer inferencias.</p>
      <p v-if="modeStatuses.length" class="mode-status-line">Modos: <span v-for="item in modeStatuses" :key="item.id">{{ item.label }} · {{ item.available ? 'disponible' : 'Modelo no disponible' }}. </span></p>
      <div class="model-switch" aria-label="Arquitecturas"><button v-for="model in models" :key="model.id" :aria-pressed="selected.id === model.id" @click="selected = model">{{ model.name }}</button></div>
      <div class="report-strip" aria-live="polite"><div class="report-note"><b>{{ selected.name }}</b><p>{{ selected.detail }}</p></div>
        <template v-if="report"><div><strong>{{ percent(report.final.accuracy) }}</strong><small>Exactitud en test</small></div><div><strong>{{ percent(report.final.recall_per_class.Fight) }}</strong><small>Peleas del test detectadas</small></div><div><strong>{{ percent(report.final.f1_macro) }}</strong><small>F1 macro</small></div></template>
        <p v-else class="report-missing">{{ loading ? 'Consultando la evaluación…' : 'Informe de test no disponible en la API.' }}</p>
      </div>
      <div class="evidence-links"><a href="#datos">Ver datos y particiones ↗</a><a href="#lote">Analizar varios videos ↗</a></div>
    </div>
  </section>
</template>
