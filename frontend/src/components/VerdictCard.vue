<script setup>
import { computed } from 'vue'
import { criticalSegment, formatTime, summarize } from '../lib/analysis.js'

const props = defineProps({
  segments: { type: Array, required: true },
  threshold: { type: Number, required: true },
  clipUrl: { type: String, default: null },
})

const summary = computed(() => summarize(props.segments, props.threshold))
const critical = computed(() => criticalSegment(props.segments))
</script>

<template>
  <section class="card verdict" :data-level="summary.level.id">
    <div>
      <p class="label">Veredicto</p>
      <p class="level">{{ summary.level.label }}</p>
      <p class="muted">
        {{ summary.flagged }} de {{ summary.total }} segmentos sospechosos
        ({{ summary.percent.toFixed(0) }} %)
      </p>
    </div>
    <div v-if="critical">
      <p class="label">Momento más crítico</p>
      <p class="level">{{ formatTime(critical.start) }}–{{ formatTime(critical.end) }}</p>
      <p class="muted">probabilidad {{ (critical.score * 100).toFixed(0) }} %</p>
    </div>
    <div class="actions">
      <a v-if="clipUrl" class="button" :href="clipUrl" download>Descargar clip</a>
      <button v-else type="button" disabled title="El clip lo genera el backend">
        Descargar clip (no disponible)
      </button>
    </div>
    <p class="note">
      Es una ayuda para priorizar la revisión humana, no una decisión final. Puede equivocarse con
      juegos, abrazos, deportes, poca luz o baja resolución.
    </p>
  </section>
</template>
