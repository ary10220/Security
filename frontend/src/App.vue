<script setup>
import { onMounted, ref } from 'vue'
import AnalizarTab from './tabs/AnalizarTab.vue'
import LoteTab from './tabs/LoteTab.vue'
import MetricasTab from './tabs/MetricasTab.vue'
import DatosTab from './tabs/DatosTab.vue'
import HistorialTab from './tabs/HistorialTab.vue'
import { useApi } from './composables/useApi.js'

const tabs = [
  { id: 'analizar', label: 'Analizar' },
  { id: 'lote', label: 'Lote' },
  { id: 'metricas', label: 'Métricas' },
  { id: 'datos', label: 'Datos' },
  { id: 'historial', label: 'Historial' },
]
const activeTab = ref('analizar')
const health = ref(null)
const api = useApi()

onMounted(async () => {
  try {
    health.value = await api.health()
  } catch {
    health.value = null
  }
})
</script>

<template>
  <div class="dashboard">
    <header class="app-header">
      <div>
        <p class="eyebrow">UAGRM · Tópicos y Programación Avanzada</p>
        <h1>Detector de comportamiento sospechoso</h1>
        <p class="muted">Análisis de video para priorizar la revisión humana.</p>
      </div>
      <div class="status-chip" :class="{ ready: health?.model_loaded }" role="status">
        <strong>{{ health?.model ?? 'mobilenet_lstm' }}</strong>
        <span>v1 · {{ !health ? 'API desconectada' : health.model_loaded ? 'Modelo cargado' : 'Modelo no disponible' }}</span>
      </div>
    </header>

    <nav class="app-tabs" aria-label="Secciones">
      <button v-for="item in tabs" :key="item.id" type="button"
        :class="{ active: activeTab === item.id }"
        :aria-current="activeTab === item.id ? 'page' : undefined"
        @click="activeTab = item.id">{{ item.label }}</button>
    </nav>

    <main>
      <AnalizarTab v-show="activeTab === 'analizar'" :model-ready="health?.model_loaded === true" />
      <LoteTab v-show="activeTab === 'lote'" :model-ready="health?.model_loaded === true" />
      <MetricasTab v-show="activeTab === 'metricas'" />
      <DatosTab v-show="activeTab === 'datos'" />
      <HistorialTab v-show="activeTab === 'historial'" />
    </main>
  </div>
</template>
