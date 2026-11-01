<script setup>
import { useHistorial } from '../composables/useHistorial.js'

const { items } = useHistorial()
const dateText = (value) => new Date(value).toLocaleString('es-BO')
</script>

<template>
  <section class="tab-content">
    <div class="section-heading"><div><p class="eyebrow">Guardado en este navegador</p><h2>Últimas 20 predicciones</h2></div></div>
    <p v-if="!items.length" class="muted">Todavía no hay predicciones.</p>
    <div v-else class="card table-wrap"><table>
      <thead><tr><th>Fecha</th><th>Video</th><th>Veredicto</th><th>Confianza</th></tr></thead>
      <tbody><tr v-for="item in items" :key="item.id">
        <td>{{ dateText(item.date) }}</td><td>{{ item.filename }}</td>
        <td><span class="badge" :class="item.verdict">{{ item.verdict === 'suspicious' ? 'SOSPECHOSO' : 'NORMAL' }}</span></td>
        <td>{{ (item.confidence * 100).toFixed(1) }} %</td>
      </tr></tbody>
    </table></div>
    <p class="muted">Solo se guardan fecha, nombre y resultado. Los videos no se conservan en el navegador.</p>
  </section>
</template>
