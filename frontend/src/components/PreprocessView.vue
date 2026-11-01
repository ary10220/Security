<script setup>
import { formatTime } from '../lib/analysis.js'

defineProps({ data: { type: Object, required: true } })
</script>

<template>
  <section class="card">
    <p class="label">{{ data.n_frames }} frames · {{ data.size[0] }}×{{ data.size[1] }} px</p>
    <ol class="frames">
      <li v-for="f in data.frames" :key="f.index">
        <img :src="f.src" :alt="`Frame ${f.index + 1}`" width="112" height="112" />
        <span class="muted">
          #{{ f.index + 1 }}<template v-if="f.time != null"> · {{ formatTime(f.time) }}</template>
        </span>
      </li>
    </ol>
  </section>
</template>
