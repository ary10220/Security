"""Evalua un checkpoint sobre el MISMO test de train_local_cpu y guarda cada clip.

train_local_cpu.py solo guarda las metricas agregadas (matriz de confusion,
F1...). Para el informe hace falta saber QUE clips se clasificaron mal, asi
que este script reconstruye exactamente el mismo test (mismos 100 clips de
val/, misma semilla) y guarda, por clip: video, etiqueta real, prediccion y
probabilidad de "Fight".

No entrena ni modifica el checkpoint: solo lo lee. Si la matriz de confusion
que imprime coincide con la de artifacts/metrics_<modelo>.json, el checkpoint
es el mismo que produjo esas metricas.

Uso (desde la raiz del repo):
    python -m scripts.eval_test_por_clip --dataset-path /ruta/a/RWF-2000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.dataset import VideoDataset
from src.models.metrics import compute_metrics
from src.models.model import MODEL_NAMES, load_checkpoint
from src.models.train import _run_epoch, _stratified_subset, set_seed

logger = logging.getLogger("eval_test_por_clip")


def parse_args() -> argparse.Namespace:
    """Lee los argumentos; los valores por defecto replican train_local_cpu.py."""
    parser = argparse.ArgumentParser(description="Guarda la prediccion de cada clip del test.")
    parser.add_argument("--model", type=str, default="mobilenet_lstm", choices=MODEL_NAMES)
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Carpeta que contiene train/ y val/ de RWF-2000.")
    parser.add_argument("--test-size", type=int, default=100,
                        help="Debe ser el mismo de la corrida original.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Debe ser la misma de la corrida original.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> int:
    """Reconstruye el test, predice cada clip y guarda el detalle en un JSON."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    artifacts = Path(args.artifacts_dir)

    # Misma semilla y mismo muestreo estratificado que train_local_cpu.py: asi
    # el subconjunto de test es identico y las metricas son comparables.
    set_seed(args.seed)
    test_dataset = VideoDataset(args.dataset_path, split="val")
    test_subset = _stratified_subset(test_dataset, args.test_size, args.seed)
    clips = [test_dataset.samples[i] for i in test_subset.indices]
    loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    # CPU a proposito: es el dispositivo de la corrida original y evita
    # diferencias numericas minimas entre CPU y GPU.
    device = torch.device("cpu")
    model = load_checkpoint(artifacts / f"model_{args.model}.pt", device="cpu")
    _, y_true, y_pred, y_score = _run_epoch(
        model, loader, nn.CrossEntropyLoss(), None, device, desc="Test por clip",
    )

    class_names = test_dataset.classes
    metrics = compute_metrics(y_true, y_pred, class_names=class_names)
    logger.info("acc=%.4f  f1_macro=%.4f  matriz=%s", metrics["accuracy"],
                metrics["f1_macro"], metrics["confusion_matrix"])

    # `_run_epoch` conserva el orden del loader (shuffle=False), por eso se
    # puede emparejar cada prediccion con su clip por posicion.
    rows = []
    for (path, _), real, pred, score in zip(clips, y_true, y_pred, y_score):
        error = "FN" if (real == 1 and pred == 0) else "FP" if (real == 0 and pred == 1) else "OK"
        rows.append({"video": Path(path).name, "y_true": real, "y_pred": pred,
                     "score": round(score, 4), "resultado": error})

    out = artifacts / f"predicciones_{args.model}_test.json"
    out.write_text(json.dumps({
        "model": args.model, "class_names": class_names, "seed": args.seed,
        "confusion_matrix": metrics["confusion_matrix"], "clips": rows,
    }, indent=2, ensure_ascii=False))
    logger.info("Detalle por clip en %s", out)

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
