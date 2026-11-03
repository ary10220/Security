<script setup>
const counts = [
  { label: 'Clips totales', value: 400 },
  { label: 'Train', value: 259 },
  { label: 'Val interno', value: 41 },
  { label: 'Test', value: 100 },
]
</script>

<template>
  <section class="tab-content">
    <div class="section-heading"><div><p class="eyebrow">RWF-2000 · subset CPU</p><h2>Datos utilizados</h2></div></div>
    <div class="data-stats"><div v-for="item in counts" :key="item.label" class="card stat">
      <strong>{{ item.value }}</strong><span>{{ item.label }}</span>
    </div></div>
    <div class="metric-grid">
      <section class="card">
        <h3>Distribución de clases</h3>
        <div class="data-bar"><span>Fight</span><div><i class="fight-bar" /></div><strong>200</strong></div>
        <div class="data-bar"><span>NonFight</span><div><i class="normal-bar" /></div><strong>200</strong></div>
        <p class="muted">Subset balanceado de 400 clips.</p>
      </section>
      <section class="card">
        <h3>Duración de los clips</h3>
        <div class="duration-chart" aria-label="400 clips de aproximadamente 5 segundos">
          <div class="duration-bar" /><span>≈ 5 s · 400 clips</span>
        </div>
        <p class="muted">El EDA del dataset completo registró videos de 5,0 s a 30 fps.</p>
      </section>
    </div>
    <section class="card">
      <h3>Peleas · RWF-2000 · corrida CPU local</h3>
      <p>RWF-2000 contiene 2.000 clips de vigilancia. Esta evaluación usa 300 clips de la partición original de entrenamiento (259 para entrenar y 41 de validación interna, separados por video fuente) y 100 clips de la partición original de validación como test.</p>
      <p>Clases: Fight / NonFight. Este checkpoint local corresponde a la corrida CPU de 100 clips de test. El número de fuentes de ese test no figura en su JSON; la comparación de 8/16/32 frames pertenece a otra corrida y no se atribuye a estos pesos.</p>
      <p>También está disponible el checkpoint ResNet + BiLSTM del Release: se evaluó con 400 clips de test de RWF-2000 completo (74,25 % de exactitud, F1 macro 74,24 % y recall Fight 76 %). Es una corrida distinta de la CPU; el selector de esta portada compara métricas y el modo Peleas de Analizar sigue usando MobileNet.</p>
      <p><a href="https://arxiv.org/abs/1911.05913" target="_blank" rel="noopener noreferrer">Cheng et al., RWF-2000 (2020)</a> · <a href="https://zenodo.org/records/15687512" target="_blank" rel="noopener noreferrer">descarga en Zenodo</a> · licencia del registro: CC BY 4.0. Uso en este proyecto: académico.</p>
    </section>
    <div class="metric-grid">
      <section class="card">
        <h3>Riesgo observable en interiores · ASHEN</h3>
        <p>Clases del checkpoint: Normal / Riesgo. Riesgo agrupa clips anotados como Robbery y Assault; no hay una clase específica «Robbery» ni validación propia de joyerías.</p>
        <p>Entrenamiento: 132 clips de 42 videos fuente. Test propio: <strong>62 clips de 18 videos fuente</strong>, sin fuentes compartidas con entrenamiento.</p>
      </section>
      <section class="card">
        <h3>Riesgo observable en exteriores · ASHEN</h3>
        <p>Clases del checkpoint: Normal / Riesgo. Entrenamiento: 22 clips de 8 videos fuente. Test propio: <strong>13 clips de 4 videos fuente</strong>.</p>
        <p><strong>No concluyente:</strong> el test es demasiado pequeño y no contiene videos Robbery de exterior. Solo demuestra el flujo de análisis.</p>
      </section>
    </div>
    <section class="card">
      <h3>Alcance de los datos</h3>
      <p>ASHEN tiene licencia CC BY-NC 4.0 para uso no comercial. Los resultados son de un prototipo académico y no validan despliegue en comercios reales. Se requiere revisar más clientes normales, teléfonos, herramientas, juguetes y escenas difíciles.</p>
    </section>
  </section>
</template>
