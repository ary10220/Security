// Extrae n frames uniformes de un video en el navegador y los reescala a
// size x size. Es una VISTA PREVIA LOCAL: imita los pasos 1-2 del pipeline
// (muestreo uniforme + resize) pero no es la salida del backend.

function seekTo(video, time) {
  return new Promise((resolve, reject) => {
    video.onseeked = () => resolve()
    video.onerror = () => reject(new Error('No se pudo leer el video'))
    video.currentTime = time
  })
}

export async function extractFramesLocal(file, n = 16, size = 224) {
  const url = URL.createObjectURL(file)
  try {
    const video = document.createElement('video')
    video.muted = true
    video.preload = 'auto'
    video.src = url
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve
      video.onerror = () => reject(new Error('No se pudo leer el video'))
    })
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = size
    const ctx = canvas.getContext('2d')
    const duration = video.duration
    const frames = []
    for (let i = 0; i < n; i++) {
      // Centro de cada una de las n franjas, para no caer justo en el ultimo instante.
      const t = ((i + 0.5) / n) * duration
      await seekTo(video, t)
      ctx.drawImage(video, 0, 0, size, size)
      frames.push({ index: i, time: t, src: canvas.toDataURL('image/jpeg', 0.8) })
    }
    return { frames, n_frames: n, size: [size, size], duration_s: duration, source: 'local' }
  } finally {
    URL.revokeObjectURL(url)
  }
}

export function readDuration(file) {
  const url = URL.createObjectURL(file)
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url)
      resolve(video.duration)
    }
    video.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('No se pudo leer el video'))
    }
    video.src = url
  })
}
