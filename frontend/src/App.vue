<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import ProductLanding from './ProductLanding.vue'
import AnalizarTab from './tabs/AnalizarTab.vue'
import LoteTab from './tabs/LoteTab.vue'
import MetricasTab from './tabs/MetricasTab.vue'
import DatosTab from './tabs/DatosTab.vue'
import HistorialTab from './tabs/HistorialTab.vue'
import { useApi } from './composables/useApi.js'
import './vigia.css'

const tabs = [
  ['analizar', 'Analizar'], ['lote', 'Lote'], ['metricas', 'Métricas'],
  ['datos', 'Datos'], ['historial', 'Historial'],
]
const readView = () => window.location.hash.slice(1) || 'inicio'
const view = ref(readView())
const inStudio = computed(() => tabs.some(([key]) => view.value === key))
const visited = ref(inStudio.value)
const health = ref(null)
const ready = computed(() => health.value?.model_loaded === true)
const currentName = computed(() => tabs.find(([key]) => key === view.value)?.[1])

// El hash permite volver a la portada con Atrás y compartir cada herramienta.
async function navigate() {
  view.value = readView()
  if (inStudio.value) visited.value = true
  await nextTick()
  const section = document.getElementById(view.value)
  if (section) section.scrollIntoView()
  else window.scrollTo(0, 0)
}
onMounted(async () => {
  window.addEventListener('hashchange', navigate)
  navigate()
  try { health.value = await useApi().health() } catch { health.value = null }
})
onUnmounted(() => window.removeEventListener('hashchange', navigate))
</script>

<template>
  <header id="inicio" class="mast"><div class="rail mast-inner">
    <a href="#inicio" class="wordmark" aria-label="Vigía, inicio"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M3 9l12 23h9L12 9zM18 9l8 15 9-15z" fill="currentColor"/></svg>Vigía<span>.</span></a>
    <nav class="top-links" aria-label="Producto"><a href="#metodo">Método</a><a href="#modelos">Modelos</a><a href="#datos">Datos</a></nav>
    <a class="top-cta" href="#analizar">Abrir consola ↗</a>
  </div></header>
  <main class="vigia-main">
    <div v-show="!inStudio"><ProductLanding /></div>
    <div v-if="visited" v-show="inStudio" class="studio rail">
      <a class="back-link" href="#inicio">← Volver a Vigía</a>
      <div class="studio-title"><div><span class="folio">CONSOLA / VIDEO</span><h1>{{ currentName }}</h1></div>
        <div class="status-chip" :class="{ ready }" role="status"><strong>{{ health?.model || 'API' }}</strong><span>{{ ready ? 'Modelo cargado' : health ? 'Sin checkpoint' : 'Sin conexión' }}</span></div>
      </div>
      <nav class="app-tabs" aria-label="Herramientas"><a v-for="([key,label], index) in tabs" :key="key" :href="'#'+key" :aria-current="view === key ? 'page' : undefined"><small>0{{ index+1 }}</small>{{ label }}</a></nav>
      <AnalizarTab v-show="view === 'analizar'" :model-ready="ready"/>
      <LoteTab v-show="view === 'lote'" :model-ready="ready"/>
      <MetricasTab v-show="view === 'metricas'"/><DatosTab v-show="view === 'datos'"/><HistorialTab v-show="view === 'historial'"/>
    </div>
  </main>
  <footer class="endbar"><div class="rail"><a href="#inicio" class="footer-brand">Vigía.</a><span>Inteligencia de video · UAGRM</span><a href="#historial">Historial ↗</a></div></footer>
</template>
