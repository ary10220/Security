# Demo de escenarios de Vigía

Vigía ayuda a un operador a **revisar tramos** de video. No predice delitos, no identifica con certeza armas reales y no llama automáticamente a la policía. El operador observa el contexto y decide qué hacer.

## Tres clasificadores

| Modo | Artefactos | Etiquetas que aprende | Estado inicial |
| --- | --- | --- | --- |
| Peleas | `model_mobilenet_lstm.pt`, `metrics_mobilenet_lstm.json` | Fight / NonFight de RWF-2000 | Existente; sus métricas solo miden peleas en RWF-2000 |
| Interior / joyería | `model_interior.pt`, `metrics_interior.json` | Riesgo / Normal en escenas interiores revisadas | Pendiente de datos y evaluación |
| Exterior | `model_exterior.pt`, `metrics_exterior.json` | Riesgo / Normal en escenas exteriores revisadas | Pendiente de datos y evaluación |

Los tres usan MobileNetV2 + BiLSTM y el mismo preprocesamiento de 16 frames. Cada modo nuevo requiere pesos propios, un manifiesto de clips y métricas de prueba propias. La API devuelve `503 Modelo no disponible` si falta alguno de esos artefactos. Hay que reiniciar FastAPI después de entrenar para recargar los checkpoints.

## Etiquetado para los dos modos nuevos

**Interior / joyería, positivo (`Riesgo`):** tramo donde se observa una amenaza o coerción dirigida a una persona, forcejeo, agresión física o toma de productos acompañada de intimidación claramente visible. Marcar el comienzo y el fin del hecho observable. La etiqueta describe la imagen; no afirma intención penal ni si un objeto es un arma auténtica.

**Interior / joyería, negativo (`Normal`):** compra y atención habitual, clientes señalando productos, entrega voluntaria, personal manipulando vitrinas, uso normal de teléfono, herramientas de trabajo, juguetes y gestos rápidos sin agresión. Incluir estas situaciones como negativos difíciles de distintas cámaras, iluminación y ángulos.

**Exterior, positivo (`Riesgo`):** agresión, persecución con confrontación, forcejeo, amenaza visible o arrebato con interacción física. Requiere evidencia visual dentro del intervalo.

**Exterior, negativo (`Normal`):** peatones y clientes normales, carreras o juegos, saludos, teléfonos, herramientas, juguetes, movimientos bruscos, deporte y discusiones sin agresión observable. No etiquetar una persona como riesgosa por apariencia, ropa o presencia en una zona.

Si el contexto es ambiguo, excluir el clip del entrenamiento y anotarlo para revisión. No etiquetar un video largo completo como positivo porque el evento sucede solo en unos segundos: el clasificador toma 16 frames de cada clip y aprendería partes normales como si fueran riesgo. Para UCF-Crime hay etiquetas débiles a nivel de video; el equipo debe mirar y recortar el intervalo antes de usarlo.

## Trabajo que puede hacer Ariany

Copiar [risk_labels.example.csv](risk_labels.example.csv) como un CSV privado junto a los videos. Agregar **una fila por intervalo revisado** con:

- `file`: ruta del video relativa al CSV.
- `source_id`: ID estable del video original. Todos los recortes del mismo original llevan el mismo ID.
- `mode`: `interior` o `exterior`.
- `label`: `Riesgo` o `Normal`.
- `start_sec`, `end_sec`: inicio y fin observados, en segundos; intervalo máximo de 30 s.
- `split`: `train` o `val`. `val` queda reservado para prueba, sin usarlo para elegir el checkpoint. La herramienta separa además 15 % de `train` por fuente para validación interna.
- `rights`: fuente y permiso conocido, o `unknown`. No asumir permiso comercial.
- `notes`: descripción visible del evento, ambigüedades, objetos confundibles o motivo de falsa alerta.

Cada fila positiva debe cubrir **un solo evento**. Si el mismo evento se recorta varias veces, no contarlo como eventos independientes en la evaluación temporal.

Buscar primero escenas de interior/joyería, junto con clientes normales y negativos difíciles. Reservar fuentes completas para prueba: ninguna fuente puede estar simultáneamente en `train` y `val`. Por cada clase se requieren al menos dos videos fuente de entrenamiento y un clip de prueba, aunque esos mínimos solo permiten ejecutar el flujo; **no bastan para sostener desempeño generalizable**. Mantener cámaras, lugares y escenas distintos en test cuando sea posible.

## Preparación y entrenamiento

Desde la raíz del proyecto, después de revisar el CSV:

```powershell
python -m scripts.prepare_risk_data --manifest C:\ruta\risk_labels.csv --output data\processed\risk_modes
python -m src.models.train --mode interior --model mobilenet_lstm --dataset-path data\processed\risk_modes\interior --epochs 5
python -m scripts.evaluate_event_timeline --dataset-root data\processed\risk_modes --mode interior
```

Para exterior, repetir entrenamiento y evaluación temporal con `--mode exterior` y su carpeta. `prepare_risk_data` valida fuentes, recorta y reencodifica los intervalos. Debe usarse una carpeta de salida vacía para no mezclar una corrida anterior. `train.py` impide la mezcla de fuentes entre train, validación interna y test en los modos nuevos. Guarda matriz de confusión, recall de clips `Riesgo`, falsas alertas y ejemplos de falsos positivos y eventos omitidos en `metrics_<modo>.json`.

La tercera orden evalúa la **decisión temporal de la API** en clips de prueba. Guarda `timeline_eval_interior.json` con matriz, falsas alertas, ejemplos de error y recall de los eventos recortados, bajo el supuesto de un evento por clip positivo. El reporte indica su denominador. No generalizarlo a videos largos completos: eso requeriría agrupar tramos por evento y medir su solapamiento con anotaciones temporales.

## Fuentes candidatas y derechos

- [UCF-Crime, página oficial](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/): incluye Robbery y Fighting entre 1900 videos largos de vigilancia; los rótulos de entrenamiento son a nivel de video. La página consultada no publica una licencia clara para explotación comercial. Tratar como material de investigación hasta obtener confirmación escrita de derechos sobre los clips usados.
- [Thermal Image Dataset for Concealed Handgun Detection, Mendeley Data](https://data.mendeley.com/datasets/b6rpgr6nrh/1): **CC BY-NC 3.0**; el autor indica que el uso comercial requiere permiso previo. Contiene video térmico de pistola oculta, teléfono y llaves. No equivale a cámaras RGB de joyería ni a una etiqueta de asalto. No incorporarlo como detector comercial sin permiso y validación específica.
- [Dangerous situation detection, Mendeley Data](https://data.mendeley.com/datasets/jhyrrgj5b6/2): **CC BY 4.0**, pero contiene imágenes de objetos y expresiones; no aporta por sí solo etiquetas temporales de eventos de robo.

Un detector de armas, si se evalúa en el futuro, sería una señal **compartida y adicional**. La apariencia de una pistola de juguete puede confundirse con la de una real. Esta demo no lo incorpora porque no hay un checkpoint evaluado con negativos difíciles ni una fuente confirmada que represente las cámaras previstas.

## Límite de tiempo y demostración

Si a los 90 minutos no hay nuevos clips bien etiquetados, dejar exterior pendiente y centrar las pruebas en peleas e interior. Si tampoco hay datos adecuados de interior, mostrarlo como **Modelo no disponible**. Para cada modo disponible, subir videos de prueba no usados en entrenamiento, revisar la matriz, las falsas alertas y los tramos señalados; registrar los errores en `notes` del manifiesto. No transferir porcentajes de RWF-2000 a robos o asaltos.
