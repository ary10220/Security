<script setup>
import PrCurve from './PrCurve.vue'

defineProps({
  metrics: { type: Object, required: true },
  threshold: { type: Number, required: true },
})
const pct = (v) => (v * 100).toFixed(1) + ' %'
</script>

<template>
  <section class="card">
    <p class="label">Modelo: {{ metrics.model }}</p>
    <div class="grid2">
      <div>
        <table>
          <thead>
            <tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th></tr>
          </thead>
          <tbody>
            <tr v-for="name in metrics.class_names" :key="name">
              <td>{{ name }}</td>
              <td>{{ pct(metrics.precision_per_class[name]) }}</td>
              <td>{{ pct(metrics.recall_per_class[name]) }}</td>
              <td>{{ pct(metrics.f1_per_class[name]) }}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr><td>Accuracy</td><td colspan="3">{{ pct(metrics.accuracy) }}</td></tr>
            <tr><td>F1 macro</td><td colspan="3">{{ pct(metrics.f1_macro) }}</td></tr>
            <tr>
              <td>Inferencia</td>
              <td colspan="3">
                {{ metrics.inference_ms == null ? 'sin medir' : metrics.inference_ms + ' ms/clip' }}
              </td>
            </tr>
          </tfoot>
        </table>
        <p class="label">Matriz de confusión</p>
        <table class="confusion">
          <thead>
            <tr><th></th><th v-for="n in metrics.class_names" :key="n">pred. {{ n }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in metrics.confusion_matrix" :key="i">
              <th>real {{ metrics.class_names[i] }}</th>
              <td v-for="(v, j) in row" :key="j" :class="{ diag: i === j }">{{ v }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <PrCurve v-if="metrics.pr_curve" :curve="metrics.pr_curve" :threshold="threshold" />
    </div>
  </section>
</template>
