<script setup>
import { computed } from 'vue'

const props = defineProps({
  curve: { type: Object, required: true },
  threshold: { type: Number, required: true },
})

const W = 280
const H = 220
const PAD = 32
const x = (r) => PAD + r * (W - PAD - 8)
const y = (p) => H - PAD - p * (H - PAD - 8)

const path = computed(() =>
  props.curve.recall
    .map((r, i) => `${i ? 'L' : 'M'}${x(r).toFixed(1)},${y(props.curve.precision[i]).toFixed(1)}`)
    .join(' '),
)

// Punto de la curva cuyo umbral es el mas cercano al del slider.
const marker = computed(() => {
  const { thresholds, recall, precision } = props.curve
  let best = 0
  thresholds.forEach((t, i) => {
    if (Math.abs(t - props.threshold) < Math.abs(thresholds[best] - props.threshold)) best = i
  })
  return { cx: x(recall[best]), cy: y(precision[best]), p: precision[best], r: recall[best] }
})
</script>

<template>
  <figure class="pr">
    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Curva precision-recall">
      <line :x1="PAD" :y1="H - PAD" :x2="W - 8" :y2="H - PAD" class="axis" />
      <line :x1="PAD" :y1="8" :x2="PAD" :y2="H - PAD" class="axis" />
      <path :d="path" class="curve" />
      <circle :cx="marker.cx" :cy="marker.cy" r="5" class="marker" />
      <text :x="W / 2" :y="H - 6" text-anchor="middle">recall</text>
      <text x="10" :y="H / 2" text-anchor="middle" :transform="`rotate(-90 10 ${H / 2})`">
        precisión
      </text>
    </svg>
    <figcaption class="muted">
      Con el umbral {{ threshold.toFixed(2) }}: precisión {{ (marker.p * 100).toFixed(0) }} %,
      recall {{ (marker.r * 100).toFixed(0) }} %. AP = {{ curve.average_precision.toFixed(3) }}
    </figcaption>
  </figure>
</template>
