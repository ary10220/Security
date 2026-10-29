<script setup>
import { computed } from 'vue'
import { flagSegments, formatTime, suspiciousRuns } from '../lib/analysis.js'

const props = defineProps({
  segments: { type: Array, required: true },
  threshold: { type: Number, required: true },
  duration: { type: Number, required: true },
  currentTime: { type: Number, default: 0 },
})
const emit = defineEmits(['seek'])

const flagged = computed(() => flagSegments(props.segments, props.threshold))
const runs = computed(() => suspiciousRuns(props.segments, props.threshold))
const pct = (t) => `${Math.min(100, (t / props.duration) * 100)}%`
const pctText = (score) => `${(score * 100).toFixed(0)} %`
</script>

<template>
  <section class="card">
    <p class="label">Línea de tiempo</p>
    <div class="timeline" role="list">
      <button
        v-for="s in flagged"
        :key="s.start"
        type="button"
        role="listitem"
        class="seg"
        :class="{ on: s.suspicious }"
        :style="{ left: pct(s.start), width: pct(s.end - s.start), '--score': s.score }"
        :title="`${formatTime(s.start)}–${formatTime(s.end)} · ${pctText(s.score)}`"
        :aria-label="`${formatTime(s.start)} a ${formatTime(s.end)}, probabilidad ${pctText(s.score)}, ${s.suspicious ? 'sospechoso' : 'normal'}`"
        @click="emit('seek', s.start)"
      />
      <span class="playhead" :style="{ left: pct(currentTime) }" />
    </div>
    <div class="ticks"><span>0:00</span><span>{{ formatTime(duration) }}</span></div>
    <ul v-if="runs.length" class="runs">
      <li v-for="r in runs" :key="r.start">
        <button type="button" class="link" @click="emit('seek', r.start)">
          {{ formatTime(r.start) }}–{{ formatTime(r.end) }}
        </button>
        <span class="muted"> pico {{ pctText(r.peak) }}</span>
      </li>
    </ul>
    <p v-else class="muted">Ningún segmento supera el umbral.</p>
  </section>
</template>
