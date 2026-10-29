<script setup>
import { ref } from 'vue'
import { ACCEPTED_TYPES, MAX_UPLOAD_MB } from '../config.js'

const props = defineProps({ disabled: Boolean })
const emit = defineEmits(['select'])

const dragging = ref(false)
const error = ref('')
const input = ref(null)

function validate(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Solo se aceptan videos .mp4.'
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `El video supera ${MAX_UPLOAD_MB} MB.`
  return ''
}

function pick(file) {
  if (!file) return
  error.value = validate(file)
  if (!error.value) emit('select', file)
}

function onDrop(event) {
  dragging.value = false
  if (!props.disabled) pick(event.dataTransfer.files[0])
}
</script>

<template>
  <div
    class="dropzone"
    :class="{ dragging, disabled }"
    role="button"
    tabindex="0"
    @click="!disabled && input.click()"
    @keydown.enter="!disabled && input.click()"
    @dragover.prevent="dragging = true"
    @dragleave="dragging = false"
    @drop.prevent="onDrop"
  >
    <input ref="input" type="file" accept="video/mp4" hidden @change="pick($event.target.files[0])" />
    <p class="title">Arrastra un video .mp4 aquí</p>
    <p class="hint">o haz clic para elegirlo (máx. {{ MAX_UPLOAD_MB }} MB)</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </div>
</template>
