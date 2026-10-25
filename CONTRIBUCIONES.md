# Contribuciones

Registro individual del trabajo de cada integrante del grupo. La idea es
que al momento de la evaluación cada una pueda señalar los commits y las
partes del código que redactó, y que el ingeniero vea con claridad el
reparto.

Reglas:

- Cada integrante llena **su propia sección**.
- Enlazar commits con el hash corto (p.ej. `a1b2c3d`) o PRs con `#NN`.
- Nombrar los archivos/módulos sobre los que se trabajó, no sólo "ayudé".
- Si se trabajó en pareja, usar `Co-authored-by:` en el commit y
  mencionarlo acá en ambas secciones.

---

## P1 — Ariany Claure

- (pendiente)

## P2 — Lorgio

- (pendiente)

## P3 — Diego

- (pendiente)

## P4 — Grageda

Fase de datos (en pareja con Diego, que se encargó de RWF-2000):

- **Real Life Violence Situations:** descarga desde Kaggle
  (`mohamedmustafa/real-life-violence-situations-dataset`) con mi cuenta.
  Verificación local: 2000 videos (1000 `Violence` + 1000 `NonViolence`),
  1.81 GB, sin archivos vacíos ni ilegibles al abrirlos con OpenCV. El zip
  de 4 GB traía el dataset duplicado (MD5 idénticos), así que se subió una
  sola copia.
- **Subida al Drive del grupo** en `datasets/real_life/`, separada de
  `rwf2000/`. Conteo verificado en el Drive: 1000 y 1000.
- **Documentación de datasets:** reescribí `data/README.md` con las fuentes,
  licencias y estructura real de RWF-2000, Real Life y Hockey Fight
  (commit `7ae8319`, PR #2).
- Los videos y el manifiesto de verificación no se versionan; viven fuera
  del repo (Drive / copia local).

Pendiente de mi parte según el plan: EDA (con Lorgio), entrenamiento (con
Ariany) y frontend (con Diego). Se registrará aquí cuando se haga.
