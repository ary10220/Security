<script setup>
import { ref } from 'vue'

const paused = ref(false)
const surface = ref(null)

// El desplazamiento acompaña al puntero sin cambiar el encuadre del diagrama.
function follow(event) {
  if (paused.value || event.pointerType === 'touch') return
  const box = surface.value.getBoundingClientRect()
  surface.value.style.setProperty('--dx', ((event.clientX - box.left) / box.width - .5) * 16 + 'px')
  surface.value.style.setProperty('--dy', ((event.clientY - box.top) / box.height - .5) * 12 + 'px')
}
function center() {
  surface.value.style.removeProperty('--dx')
  surface.value.style.removeProperty('--dy')
}
</script>

<template>
  <figure ref="surface" class="optical" :class="{ paused }" @pointermove="follow" @pointerleave="center">
    <div class="plate-label">FIG. 01 <span>/</span> ÓPTICA</div>
    <svg class="optic-svg" viewBox="0 0 560 440" role="img" aria-label="Ilustración interactiva de una lente óptica con anillos de enfoque">
      <defs>
        <radialGradient id="halo"><stop stop-color="#98b363" stop-opacity=".41"/><stop offset=".64" stop-color="#5e7353" stop-opacity=".22"/><stop offset="1" stop-color="#29352d" stop-opacity="0"/></radialGradient>
        <linearGradient id="rim" x2=".73" y2="1"><stop stop-color="#e8eed6"/><stop offset=".11" stop-color="#6d776b"/><stop offset=".29" stop-color="#29312c"/><stop offset=".53" stop-color="#080f0c"/><stop offset=".77" stop-color="#596355"/><stop offset=".93" stop-color="#c1c7ad"/><stop offset="1" stop-color="#465148"/></linearGradient>
        <radialGradient id="glass" cx=".35" cy=".27" r=".86"><stop stop-color="#b0c996"/><stop offset=".16" stop-color="#687c69"/><stop offset=".32" stop-color="#334a44"/><stop offset=".54" stop-color="#0c2020"/><stop offset=".77" stop-color="#182e28"/><stop offset=".91" stop-color="#596e48"/><stop offset="1" stop-color="#0b1712"/></radialGradient>
        <linearGradient id="blade" x2="1" y2=".8"><stop stop-color="#101d16"/><stop offset=".45" stop-color="#506052"/><stop offset="1" stop-color="#9eac82"/></linearGradient>
        <linearGradient id="glint" x2=".8" y2="1"><stop stop-color="#f2ffd6" stop-opacity=".62"/><stop offset=".6" stop-color="#a3c5ab" stop-opacity=".11"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
        <radialGradient id="pupil" cx=".35" cy=".24" r=".8"><stop stop-color="#536765"/><stop offset=".33" stop-color="#142727"/><stop offset=".7" stop-color="#020907"/><stop offset="1" stop-color="#263b2a"/></radialGradient>
        <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".85" fill="#d9e4cb" opacity=".24"/></pattern>
        <filter id="shadow" x="-50%" y="-70%" width="200%" height="240%"><feGaussianBlur stdDeviation="10"/></filter>
      </defs>
      <rect width="560" height="440" fill="url(#grid)"/>
      <ellipse cx="300" cy="206" rx="262" ry="215" fill="url(#halo)"/>
      <g fill="none" stroke="#a8b99b" opacity=".36" stroke-width=".7">
        <ellipse cx="283" cy="218" rx="240" ry="158" transform="rotate(-24 283 218)" stroke-dasharray="3 6"/>
        <path d="M26 230h82m347 0h81M286 17v42m0 316v49M40 74h9m-4.5-4.5v9m459 255h9m-4.5-4.5v9"/>
      </g>
      <ellipse cx="291" cy="364" rx="146" ry="17" fill="#030704" opacity=".65" filter="url(#shadow)"/>
      <g class="optic-object">
        <g transform="translate(285 207) rotate(-19) scale(1 .91)">
          <circle cy="20" r="157" fill="#080e0b" stroke="#54604f" stroke-width="2"/>
          <circle v-for="j in 5" :key="'base'+j" :cy="20-j*3" r="156" fill="#19201c" stroke="#475044" stroke-width="1"/>
          <circle r="159" fill="url(#rim)"/><circle r="154" fill="#18211b"/>
          <g stroke="#98a491"><path v-for="j in 156" :key="j" :transform="'rotate('+j*360/156+')'" d="M0 -152l.8 7" stroke-width="1.15"/></g>
          <circle r="142" fill="url(#rim)"/><circle r="137" fill="#141c17"/>
          <circle v-for="r in [134,130,126,123]" :key="r" :r="r" fill="none" stroke="#5f6c5a" stroke-width="1"/>
          <circle r="119" fill="url(#glass)" stroke="#899478" stroke-width="1.5"/>
          <g class="iris-blades"><path v-for="j in 9" :key="j" :transform="'rotate('+40*j+')'" d="M0 -106 Q47 -105 69 -76 L36 -18 Q14 -57 -33 -46Z" fill="url(#blade)" opacity=".78" stroke="#172821" stroke-width="1"/></g>
          <circle r="49" fill="url(#pupil)" stroke="#8a9c76" stroke-opacity=".55"/>
          <ellipse cx="-40" cy="-66" rx="59" ry="14" fill="url(#glint)" transform="rotate(-16)"/>
          <path d="M-101 -19 A104 104 0 0 1 43 -94" fill="none" stroke="#e1eccb" opacity=".54" stroke-width="2"/>
          <path d="M101 30 A106 106 0 0 1 -37 99" fill="none" stroke="#a3cc83" opacity=".75" stroke-width="3"/>
          <ellipse cx="-15" cy="-20" rx="13" ry="4" fill="#d9f4de" opacity=".56" transform="rotate(-20)"/>
        </g>
      </g>
      <g fill="none" stroke="#c3eb88" stroke-width="1.6"><path d="M210 173v-10h11m127 0h11v10m0 80v10h-11m-127 0h-11v-10"/></g>
      <g class="optic-sweep"><path d="M190 215h189" stroke="#d0ff97" stroke-width="1" opacity=".82"/></g>
    </svg>
    <figcaption><i></i> Ilustración del enfoque</figcaption>
    <button class="motion-toggle" type="button" :aria-label="paused ? 'Reanudar animación' : 'Pausar animación'" :aria-pressed="paused" @click="paused = !paused; center()">{{ paused ? '▶' : 'Ⅱ' }}</button>
  </figure>
</template>
