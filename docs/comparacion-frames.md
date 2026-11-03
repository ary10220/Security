# Comparación de 8, 16 y 32 frames por clip (2 de noviembre de 2026)

Cierra la decisión abierta "¿se confirma 16 frames?": hasta ahora los notebooks solo medían cobertura y costo, no exactitud.

## Cómo se midió
- Modelo `mobilenet_lstm`, RWF-2000 completo, el mismo comando y split que las corridas de GPU del PR #32: 1.393 clips de train, 207 de validación interna (por fuente) y los 400 de `val/` como test. Semilla 42, batch 8, hasta 15 épocas con early stopping (paciencia 5).
- Lo único que cambia es `--num-frames` (PR #33), aplicado igual a train, validación interna y test.
- GPU: GTX 1050 Ti (4 GB), una sola corrida por configuración. Los tiempos están afectados porque en paralelo se descargaban videos y se hacían otros análisis.

## Resultados (test: 400 clips, 200 por clase)

| Frames | Exactitud | IC 95 % (Wilson) | F1 macro | Matriz `[[NonFight→NonFight, NonFight→Fight], [Fight→NonFight, Fight→Fight]]` | Mejor época / corridas | Inferencia por clip* | Entrenamiento |
|---|---|---|---|---|---|---|---|
| 8 | 72,50 % | 67,9–76,6 % | 0,725 | `[[148, 52], [58, 142]]` | 6 / 11 | 60,9 ms | 23,4 min |
| 16 | 75,25 % | 70,8–79,2 % | 0,752 | `[[145, 55], [44, 156]]` | 2 / 7 | 102,0 ms | 25,1 min |
| 32 | 74,00 % | 69,5–78,1 % | 0,740 | `[[156, 44], [60, 140]]` | 6 / 11 | 257,1 ms | 73,8 min |

\* Incluye decodificar el video, como en `metrics_*.json`.

Recall de Fight: 71,0 % (8), 78,0 % (16) y 70,0 % (32). Los archivos originales están en `artifacts/frames/`.

## Qué se puede concluir
- **Las diferencias no son distinguibles del ruido.** Los intervalos de confianza se solapan casi por completo. Además, la corrida de 16 frames de Diego (misma configuración y semilla, otra GPU) dio 73,0 %, y esta dio 75,25 %: la variación entre corridas ya es de 2,25 puntos, comparable a la diferencia entre 8, 16 y 32 (hasta 2,75 puntos).
- **El costo sí es claro.** 32 frames multiplica por ~2,5 el tiempo de inferencia por clip respecto a 16 y por ~3 el de entrenamiento, sin mejora medible. 8 frames es más rápido pero pierde recall de Fight en esta corrida, también dentro del ruido.
- **Decisión razonable:** mantener 16 frames (el valor del plan) por costo y cobertura, no porque esta comparación demuestre que sea mejor. Esto vale solo para `mobilenet_lstm` en RWF-2000.
- **Todas quedan en torno a la regla de resolución** (75,0 % en los mismos 400 clips, ver `docs/auditoria-ashen.md` y `scripts/analisis_resolucion.py`). Ningún valor de frames cambia esa conclusión.

## Límites
Una sola semilla por configuración, 400 clips de test y un solo modelo. Para afirmar una diferencia haría falta repetir con varias semillas y comparar los mismos clips (por ejemplo con una prueba de McNemar). No se evaluó el efecto en Real Life ni en los modos interior y exterior.
