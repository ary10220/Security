<script setup>
defineProps({
  modelValue: { type: Number, required: true },
  bestThreshold: { type: Number, default: null },
})
defineEmits(['update:modelValue'])
</script>

<template>
  <section class="card">
    <label class="label" for="threshold">
      Umbral de decisión: <strong>{{ modelValue.toFixed(2) }}</strong>
    </label>
    <input
      id="threshold"
      type="range"
      min="0"
      max="1"
      step="0.01"
      :value="modelValue"
      @input="$emit('update:modelValue', Number($event.target.value))"
    />
    <p class="muted">
      Un umbral bajo detecta más casos (sube el recall) pero genera más falsas alarmas; uno alto
      hace lo contrario.
      <button
        v-if="bestThreshold !== null"
        type="button"
        class="link"
        @click="$emit('update:modelValue', bestThreshold)"
      >
        Usar el que maximiza F1 en validación ({{ bestThreshold.toFixed(2) }})
      </button>
    </p>
  </section>
</template>
