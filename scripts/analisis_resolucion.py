"""Compara los modelos de peleas contra la regla de resolucion, clip por clip.

El EDA (notebooks/02_eda_datasets.ipynb) encontro que una regla de un solo
umbral sobre el alto del video separa Fight/NonFight en RWF-2000 con ~75 % de
exactitud. Si los modelos no la superan, su exactitud no prueba que hayan
aprendido el comportamiento. Este script usa los CSV de
`scripts/eval_test_por_clip.py` (mismos 400 clips de val/ para los tres
modelos) y mide dos cosas:

    1. Exactitud de la regla y de cada modelo en esos clips.
    2. Exactitud por resolucion, junto al acierto de la respuesta "siempre la
       clase mayoritaria" de ese grupo. Un modelo que no supera esa linea no
       esta aportando nada dentro del grupo.

Uso (desde la raiz del repo):
    python -m scripts.analisis_resolucion
    python -m scripts.analisis_resolucion --alto-umbral 251 --salida artifacts/analisis_resolucion.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

MODELOS = ("mobilenet_baseline", "mobilenet_lstm", "resnet_lstm")
# Umbral fijado en train/ por el EDA (notebook 02, seccion 6): alto > 251 -> Fight.
ALTO_UMBRAL = 251


def leer_predicciones(path: Path) -> list[dict]:
    """Lee un CSV de `eval_test_por_clip.py`.

    Args:
        path: ruta a `predicciones_<modelo>_test.csv`.

    Returns:
        Filas con `y_true`, `y_pred`, `ancho` y `alto` ya convertidos a int.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for campo in ("y_true", "y_pred", "ancho", "alto"):
            row[campo] = int(float(row[campo]))
    return rows


def exactitud(aciertos: int, total: int) -> float:
    """Proporcion de aciertos; 0 si no hay clips (evita dividir entre cero)."""
    return aciertos / total if total else 0.0


def analizar(preds: dict[str, list[dict]], alto_umbral: int = ALTO_UMBRAL) -> dict:
    """Calcula la exactitud de la regla y la de cada modelo, global y por resolucion.

    Args:
        preds: filas de predicciones por nombre de modelo. Todos los modelos
            deben tener los mismos clips en el mismo orden.
        alto_umbral: la regla predice Fight (1) si el alto supera este valor.

    Returns:
        Diccionario con los resultados globales y por resolucion.

    Raises:
        ValueError: si los modelos no comparten exactamente los mismos clips,
            porque entonces la comparacion clip por clip no tendria sentido.
    """
    referencia = preds[next(iter(preds))]
    videos = [row["video"] for row in referencia]
    for nombre, rows in preds.items():
        if [row["video"] for row in rows] != videos:
            raise ValueError(f"{nombre} no tiene los mismos clips que los demas modelos.")

    y_true = [row["y_true"] for row in referencia]
    regla = [int(row["alto"] > alto_umbral) for row in referencia]
    resultado: dict = {
        "n_clips": len(y_true),
        "alto_umbral": alto_umbral,
        "regla_exactitud": exactitud(sum(r == y for r, y in zip(regla, y_true)), len(y_true)),
        "modelos": {},
        "por_resolucion": [],
    }
    for nombre, rows in preds.items():
        y_pred = [row["y_pred"] for row in rows]
        resultado["modelos"][nombre] = {
            "exactitud": exactitud(sum(p == y for p, y in zip(y_pred, y_true)), len(y_true)),
            "acuerdo_con_regla": exactitud(sum(p == r for p, r in zip(y_pred, regla)), len(y_true)),
        }

    grupos: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(referencia):
        grupos[(row["ancho"], row["alto"])].append(i)
    for (ancho, alto), indices in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
        etiquetas = [y_true[i] for i in indices]
        mayoritaria = Counter(etiquetas).most_common(1)[0][1]
        resultado["por_resolucion"].append({
            "resolucion": f"{ancho}x{alto}",
            "n": len(indices),
            "proporcion_fight": exactitud(sum(etiquetas), len(indices)),
            # Acierto de contestar siempre la clase mayoritaria del grupo.
            "linea_mayoritaria": exactitud(mayoritaria, len(indices)),
            "exactitud_modelos": {
                nombre: exactitud(
                    sum(rows[i]["y_pred"] == y_true[i] for i in indices), len(indices))
                for nombre, rows in preds.items()
            },
        })
    return resultado


def main() -> int:
    """Lee los CSV de los tres modelos, imprime el resumen y guarda el JSON."""
    parser = argparse.ArgumentParser(description="Regla de resolucion contra los modelos.")
    parser.add_argument("--artifacts", type=str, default="artifacts",
                        help="Carpeta con predicciones_<modelo>_test.csv.")
    parser.add_argument("--alto-umbral", type=int, default=ALTO_UMBRAL,
                        help="La regla predice Fight si el alto supera este valor.")
    parser.add_argument("--salida", type=str, default="artifacts/analisis_resolucion.json",
                        help="Donde guardar el resultado.")
    args = parser.parse_args()

    carpeta = Path(args.artifacts)
    preds = {m: leer_predicciones(carpeta / f"predicciones_{m}_test.csv") for m in MODELOS}
    resultado = analizar(preds, args.alto_umbral)

    print(f"{resultado['n_clips']} clips. Regla alto>{args.alto_umbral}: "
          f"{resultado['regla_exactitud']:.3f}")
    for nombre, datos in resultado["modelos"].items():
        print(f"  {nombre}: exactitud {datos['exactitud']:.3f}, "
              f"acuerdo con la regla {datos['acuerdo_con_regla']:.3f}")
    for grupo in resultado["por_resolucion"][:5]:
        modelos = " ".join(f"{m[:9]}={v:.2f}" for m, v in grupo["exactitud_modelos"].items())
        print(f"  {grupo['resolucion']:>9} n={grupo['n']:3d} "
              f"mayoritaria={grupo['linea_mayoritaria']:.2f} | {modelos}")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Guardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
