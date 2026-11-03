# Auditoría de las etiquetas de ASHEN (2 de noviembre de 2026)

Revisión de `data/ashen/videos.csv` (78 videos, PR #32). Hecha con asistencia de Claude Code; no sustituye una revisión humana de todos los videos.

## Ambiente interior / exterior
- Se descargaron los 78 videos y se clasificaron a ciegas, con 2 fotogramas por video, **24 videos**: los 12 exteriores y 12 interiores elegidos al azar (semilla 7).
- Coinciden **23 de 24**. El dudoso es `Normal_Videos065_x264.mp4` (vista de puertas de vidrio de un vestíbulo, no queda claro si es interior).
- Límites: la revisión la hizo un asistente y no una persona; no se vieron los otros 48 interiores.

## Separación train/test
- `manifest.json`: ninguna fuente aparece a la vez en train y val. Interior: 42 fuentes train y 18 val (132 y 62 clips). Exterior: 8 y 4 fuentes (22 y 13 clips).

## Posible atajo por metadatos (a nivel de video)
Umbral de un solo atributo elegido en los videos de train y evaluado en los de test:

| Ambiente | Test (videos) | Mejor acierto en test |
|---|---|---|
| Interior | 18 (9 y 9) | 56–61 % |
| Exterior | 4 (2 y 2) | 50–75 % |
| Todos | 22 (11 y 11) | 55–64 % |

No es un atajo fuerte, pero con tan pocos videos de test no se puede descartar.

## Sesgo de selección
- Los 40 normales se eligieron como los más livianos de `test_normal.txt`. Todos son 320x240, con mediana de 6,4 MB y 35 s. Los videos de Riesgo tienen mediana de 10,8 MB y 63 s, y 10 de 37 son 1280x720.
- No se comprobó si `prepare_risk_data` iguala la resolución al recortar.

## Licencia
ASHEN es CC BY-NC 4.0: uso académico, no comercial.
