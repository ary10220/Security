"""Evalua un checkpoint sobre su test y guarda la prediccion de cada clip.

train_local_cpu.py y train.py solo guardan metricas agregadas (matriz de
confusion, F1...). Para el informe y para comparar contra la regla de
resolucion hace falta saber QUE clips se clasificaron mal, asi que este
script reconstruye el test y guarda, por clip: video, fuente, etiqueta real,
prediccion, probabilidad de la clase positiva y resolucion del video.

Dos tests posibles:
    - Por defecto, el MISMO subset de train_local_cpu.py (100 clips de val/,
      misma semilla), para la corrida en CPU.
    - Con --test-size 0, val/ completo: es el test de src/models/train.py
      (400 clips en RWF-2000, o la carpeta val/ de un modo de riesgo).

No entrena ni modifica el checkpoint: solo lo lee. Si la matriz de confusion
que imprime coincide con la de artifacts/metrics_<artefacto>.json, el
checkpoint es el mismo que produjo esas metricas; si no coincide, avisa.

Uso (desde la raiz del repo):
    python -m scripts.eval_test_por_clip --dataset-path /ruta/a/RWF-2000
    python -m scripts.eval_test_por_clip --dataset-path /ruta/a/RWF-2000 \
        --test-size 0 --device cuda --model resnet_lstm
    python -m scripts.eval_test_por_clip --dataset-path /ruta/risk_modes/interior \
        --test-size 0 --device cuda --mode interior
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.models.dataset import VideoDataset, source_id
from src.models.metrics import compute_metrics
from src.models.model import MODEL_NAMES, load_checkpoint
from src.models.train import _run_epoch, _stratified_subset, set_seed

logger = logging.getLogger("eval_test_por_clip")

# Mismas etiquetas que usa train.py para los modos de riesgo: las carpetas se
# llaman Normal y Riesgo, y Riesgo es la clase positiva.
RISK_LABEL_MAP = {"Normal": 0, "Riesgo": 1}
MODES = ("peleas", "interior", "exterior")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos; los valores por defecto replican train_local_cpu.py."""
    parser = argparse.ArgumentParser(description="Guarda la prediccion de cada clip del test.")
    parser.add_argument("--model", type=str, default="mobilenet_lstm", choices=MODEL_NAMES)
    parser.add_argument("--mode", type=str, default="peleas", choices=MODES,
                        help="Modo del checkpoint; los de riesgo usan model_<modo>.pt.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Carpeta que contiene train/ y val/.")
    parser.add_argument("--test-size", type=int, default=100,
                        help="Clips del subset de la corrida en CPU; 0 = val/ completo.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Debe ser la misma de la corrida original.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"),
                        help="Usar el mismo dispositivo con el que se midio el JSON.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    return parser.parse_args()


def artifact_name(model: str, mode: str) -> str:
    """Nombre de los artefactos, con la misma regla que src/models/train.py.

    Args:
        model: arquitectura (uno de MODEL_NAMES).
        mode: peleas, interior o exterior.

    Returns:
        El nombre del modelo en peleas, o el del modo en los modos de riesgo.
    """
    return model if mode == "peleas" else mode


def label_map_for(mode: str) -> dict[str, int] | None:
    """Mapa de etiquetas del modo; None deja que VideoDataset detecte Fight/NonFight.

    Args:
        mode: peleas, interior o exterior.

    Returns:
        RISK_LABEL_MAP para los modos de riesgo, None para peleas.
    """
    return None if mode == "peleas" else dict(RISK_LABEL_MAP)


def select_test_indices(dataset: VideoDataset, test_size: int, seed: int) -> list[int]:
    """Indices de los clips que forman el test.

    Args:
        dataset: val/ completo.
        test_size: 0 para usar todo val/ (test de train.py); otro valor para el
            subset estratificado de train_local_cpu.py.
        seed: semilla de la corrida original.

    Returns:
        Indices sobre `dataset`, en el orden en que se evaluan.
    """
    if test_size == 0:
        return list(range(len(dataset)))
    return list(_stratified_subset(dataset, test_size, seed).indices)


def video_resolution(path: str | Path) -> tuple[int, int]:
    """Ancho y alto del video, para comparar contra la regla de resolucion del EDA.

    Args:
        path: ruta del video.

    Returns:
        (ancho, alto); (0, 0) si OpenCV no puede abrirlo.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def main() -> int:
    """Reconstruye el test, predice cada clip y guarda el detalle en JSON y CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    artifacts = Path(args.artifacts_dir)
    name = artifact_name(args.model, args.mode)

    # Misma semilla que la corrida original: con --test-size distinto de 0 el
    # subset estratificado es identico al de train_local_cpu.py.
    set_seed(args.seed)
    test_dataset = VideoDataset(args.dataset_path, split="val",
                                label_map=label_map_for(args.mode))
    indices = select_test_indices(test_dataset, args.test_size, args.seed)
    clips = [test_dataset.samples[i] for i in indices]
    loader = DataLoader(Subset(test_dataset, indices), batch_size=args.batch_size,
                        shuffle=False)

    # Por defecto CPU, el dispositivo de la corrida original; las corridas de
    # GPU se reevaluan en cuda para evitar diferencias numericas minimas.
    device = torch.device(args.device)
    model = load_checkpoint(artifacts / f"model_{name}.pt", device=args.device)
    _, y_true, y_pred, y_score = _run_epoch(
        model, loader, nn.CrossEntropyLoss(), None, device, desc="Test por clip",
    )

    class_names = test_dataset.classes
    metrics = compute_metrics(y_true, y_pred, class_names=class_names)
    logger.info("acc=%.4f  f1_macro=%.4f  matriz=%s", metrics["accuracy"],
                metrics["f1_macro"], metrics["confusion_matrix"])

    # Control: la matriz debe ser la misma que la del JSON de la corrida. Si
    # no, el checkpoint, el test o el dispositivo no son los mismos.
    metrics_path = artifacts / f"metrics_{name}.json"
    if metrics_path.is_file():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))["final"]["confusion_matrix"]
        if saved == metrics["confusion_matrix"]:
            logger.info("La matriz coincide con %s.", metrics_path)
        else:
            logger.warning("La matriz NO coincide con %s (%s).", metrics_path, saved)

    # `_run_epoch` conserva el orden del loader (shuffle=False), por eso se
    # puede emparejar cada prediccion con su clip por posicion.
    rows = []
    for (path, _), real, pred, score in zip(clips, y_true, y_pred, y_score):
        error = "FN" if (real == 1 and pred == 0) else "FP" if (real == 0 and pred == 1) else "OK"
        width, height = video_resolution(path)
        rows.append({"video": Path(path).name, "source_id": source_id(path),
                     "clase_real": class_names[real], "y_true": real, "y_pred": pred,
                     "score": round(score, 4), "resultado": error,
                     "ancho": width, "alto": height})

    out = artifacts / f"predicciones_{name}_test.json"
    out.write_text(json.dumps({
        "model": args.model, "mode": args.mode, "class_names": class_names,
        "seed": args.seed, "test_size": len(rows),
        "confusion_matrix": metrics["confusion_matrix"], "clips": rows,
    }, indent=2, ensure_ascii=False))
    # CSV para revisar en una hoja de calculo y cruzar con la regla de resolucion.
    out_csv = artifacts / f"predicciones_{name}_test.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Detalle por clip en %s y %s", out, out_csv)

    # Los errores mas "seguros" (score lejos de 0.5) son los mejores ejemplos
    # para el informe: el modelo se equivoco con conviccion.
    for tipo in ("FN", "FP"):
        errores = sorted((r for r in rows if r["resultado"] == tipo),
                         key=lambda r: abs(r["score"] - 0.5), reverse=True)
        logger.info("%s: %d clips; los 3 mas seguros: %s", tipo, len(errores),
                    [(r["video"], r["score"]) for r in errores[:3]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
