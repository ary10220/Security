<script setup>
import { nextTick, onBeforeUnmount, ref } from 'vue'
import { WEBCAM_SECONDS } from '../config.js'
import { pickRecordingType, recordingFile } from '../lib/webcam.js'

defineProps({ disabled: Boolean })
const emit = defineEmits(['select'])

const preview = ref(null)
const stream = ref(null)
const recording = ref(false)
const remaining = ref(WEBCAM_SECONDS)
const error = ref('')

let recorder = null
let timer = null
let type = null
let discard = false

// Apaga la cámara (la luz del equipo debe apagarse apenas termina la grabación).
function closeCamera() {
  clearInterval(timer)
  stream.value?.getTracks().forEach((track) => track.stop())
  stream.value = null
  recording.value = false
  remaining.value = WEBCAM_SECONDS
}

async function open() {
  error.value = ''
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    error.value = 'Tu navegador no permite usar la webcam.'
    return
  }
  // Se revisa el formato antes de pedir permiso: si no se puede grabar en un
  // formato que acepte el análisis, no tiene sentido encender la cámara.
  type = pickRecordingType((t) => MediaRecorder.isTypeSupported(t))
  if (!type) {
    error.value = 'Tu navegador no graba en un formato compatible (mp4). Prueba con Chrome o sube el video.'
    return
  }
  try {
    stream.value = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  } catch (e) {
    error.value =
      e.name === 'NotAllowedError'
        ? 'Permiso de cámara denegado. Habilítalo en el navegador para grabar.'
        : 'No se encontró una cámara disponible.'
    return
  }
  await nextTick()
  preview.value.srcObject = stream.value
}

function start() {
  const chunks = []
  discard = false
  recorder = new MediaRecorder(stream.value, { mimeType: type })
  recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data)
  recorder.onstop = () => {
    closeCamera()
    if (!discard) emit('select', recordingFile(chunks, type))
  }
  recorder.start()
  recording.value = true
  remaining.value = WEBCAM_SECONDS
  timer = setInterval(() => {
    remaining.value -= 1
    if (remaining.value <= 0) recorder.stop()
  }, 1000)
}

function cancel() {
  discard = true
  if (recording.value) recorder.stop()
  else closeCamera()
}

onBeforeUnmount(cancel)
</script>

<template>
  <section class="webcam">
    <button v-if="!stream" type="button" :disabled="disabled" @click="open">Usar webcam</button>
    <template v-else>
      <video ref="preview" autoplay muted playsinline aria-label="Vista previa de la webcam" />
      <p class="muted" role="status">
        {{ recording ? 'Grabando…' : `Se grabarán ${WEBCAM_SECONDS} segundos.` }}
        <span v-if="recording" aria-hidden="true">{{ remaining }} s</span>
      </p>
      <div class="actions">
        <button v-if="!recording" type="button" class="button" @click="start">
          Grabar {{ WEBCAM_SECONDS }} s
        </button>
        <button type="button" @click="cancel">Cancelar</button>
      </div>
    </template>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </section>
</template>

<style scoped>
.webcam { margin-top: 12px; }
.webcam video { display: block; width: 100%; max-width: 480px; border-radius: 8px; background: #000; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
</style>
