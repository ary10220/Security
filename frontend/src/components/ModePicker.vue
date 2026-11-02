<script setup>
defineProps({
  modes: { type: Array, required: true },
  modelValue: { type: String, required: true },
  group: { type: String, required: true },
})
defineEmits(['update:modelValue'])

const fallback = [
  { id: 'peleas', label: 'Peleas · RWF-2000', available: false, detail: 'API sin conexión.' },
  { id: 'interior', label: 'Interior / joyería', available: false, detail: 'API sin conexión.' },
  { id: 'exterior', label: 'Exterior', available: false, detail: 'API sin conexión.' },
]
</script>

<template>
  <fieldset class="mode-picker">
    <legend>Elegir escenario de revisión</legend>
    <div class="mode-options">
      <label v-for="item in modes.length ? modes : fallback" :key="item.id"
        class="mode-option" :class="{ chosen: modelValue === item.id, unavailable: !item.available }">
        <input type="radio" :name="group" :value="item.id" :checked="modelValue === item.id"
          @change="$emit('update:modelValue', item.id)" />
        <span><strong>{{ item.label }}</strong><small>{{ item.available ? 'Modelo disponible' : 'Modelo no disponible' }}</small></span>
      </label>
    </div>
    <p class="muted">Cada escenario usa su propio checkpoint y evaluación. Las métricas de RWF-2000 corresponden solo a peleas.</p>
  </fieldset>
</template>
