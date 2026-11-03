# Datos ASHEN para los modos interior y exterior

Los modos `interior` y `exterior` se entrenaron con 78 videos del
[dataset ASHEN](https://huggingface.co/datasets/mtastan/ASHEN-Dataset)
(videos de vigilancia de UCF-Crime, reorganizados y con anotación temporal
revisada). Licencia **CC BY-NC 4.0**: uso académico, no comercial. Los videos
no se suben al repositorio.

## Archivos

| Archivo | Contenido |
|---|---|
| `videos.csv` | Una fila por video: ruta en Hugging Face, clase (`Robbery`, `Assault`, `Normal`), ambiente (`interior`, `exterior`, `excluido`), split (`train`/`test`) y motivo de exclusión. Es la fuente de verdad de las decisiones humanas. |
| `anotaciones_temporales_ashen.txt` | Archivo oficial de ASHEN (`splits_v2_native/Temporal_Anomaly_Annotation_for_Testing_Videos.txt`): inicio y fin del evento en fotogramas. |
| `risk_labels.csv` | Una fila por clip de 5 s, en el formato de `scripts/prepare_risk_data.py`. Se regenera con `scripts/build_ashen_manifest.py`. |
| `manifest.json` | Salida de `prepare_risk_data`: a qué video original pertenece cada clip recortado (los clips se llaman `<hash>_<n>.mp4`). |

## Cómo se decidió

- **Videos:** los 38 Robbery y Assault que tienen intervalo anotado (son del split de test de ASHEN) y los 40 normales más livianos de `test_normal.txt`. Fighting se dejó afuera para no duplicar el modo `peleas`.
- **Clase:** Riesgo = Robbery + Assault; Normal = videos normales.
- **Ambiente:** se clasificó mirando un fotograma del medio del evento (o dos del video normal). 6 videos quedaron excluidos por ambiguos o por no ser vigilancia; el motivo está en `videos.csv`.
- **Clips:** ventanas de 5 s dentro del evento anotado (máximo 6 por video, repartidas) y hasta 3 por video normal.
- **Split:** por video fuente, nunca por clip. Por cada ambiente y clase se sortearon con semilla 42 el 30 % de las fuentes para test (mínimo 2).

Resultado: interior 31 fuentes de Riesgo y 29 Normal; exterior 6 y 6.

## Límites

- Exterior tiene solo 4 fuentes en test (13 clips): **no se reporta ninguna exactitud** para ese modo; el flujo solo sirve para mostrar el proceso.
- Se usan videos del split de test de ASHEN como datos propios; los números **no** son comparables con los publicados para ASHEN.
- **Sesgo de selección en los normales.** Los 40 normales se eligieron como los más livianos de `test_normal.txt`. Tras las exclusiones quedan 35 normales (todos 320x240, mediana de 6,4 MB y 35 s) frente a 37 de riesgo (26 de 320x240, 10 de 1280x720 y 1 de 640x480; mediana de 10,8 MB y 63 s). `prepare_risk_data` conserva la resolución original, así que el modelo puede aprender esa diferencia en vez del comportamiento. Un umbral de un solo atributo del video acierta 55–64 % (revisión del PR #32): no es un atajo fuerte, pero con 18 videos de test no se descarta.
- **Auditoría de etiquetas:** en la revisión del PR #32 se clasificaron a ciegas 24 de los 78 videos (los 12 exteriores y 12 interiores al azar) y coincidieron 23. La hizo un asistente de IA con 2 fotogramas por video, no una persona, y no cubre los otros 48 interiores.

## Reproducir

Desde la raíz del repo, con los videos en `../ashen/` (la carpeta debe quedar con `anomaly/` y `normal/`):

```bash
# 1. Descargar los 78 videos (~0,9 GB)
cut -d, -f2 data/ashen/videos.csv | tail -n +2 | while read p; do
  mkdir -p "../ashen/$(dirname "$p")"
  curl -sfL -o "../ashen/$p" "https://huggingface.co/datasets/mtastan/ASHEN-Dataset/resolve/main/$p"
done

# 2. Etiquetas por clip y recorte (la carpeta de salida debe estar vacía)
python -m scripts.build_ashen_manifest --videos-root ../ashen --output ../ashen/risk_labels.csv
python -m scripts.prepare_risk_data --manifest ../ashen/risk_labels.csv --output ../ashen/risk_modes

# 3. Entrenar, evaluar y guardar las predicciones por clip (por modo)
python -m src.models.train --mode interior --model mobilenet_lstm \
  --dataset-path ../ashen/risk_modes/interior --device cuda --batch-size 8 --num-workers 8 --epochs 15
python -m scripts.evaluate_event_timeline --dataset-root ../ashen/risk_modes --mode interior
python -m scripts.eval_test_por_clip --dataset-path ../ashen/risk_modes/interior \
  --mode interior --test-size 0 --device cuda --batch-size 8
```

Para `exterior`, repetir el paso 3 cambiando `interior` por `exterior`.
