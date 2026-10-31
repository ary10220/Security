"""Entrenamiento local en CPU con un subset reducido de RWF-2000.

Sirve para tener un modelo y metricas reales sin GPU: con el dataset entero
una epoca en CPU tarda horas. Reusa las piezas de src/models/train.py (mismo
modelo, mismo pipeline, mismo split por fuente, mismas metricas) y solo
cambia el tamano de los datos y los hiperparametros.

Particion, igual que train.py pero sobre el subset:
    - 300 clips de train/ (mitad Fight, mitad NonFight, seed=42). De esos,
      el 15 % por video fuente es el val interno: elige el mejor checkpoint
      y dispara el early stopping.
    - 100 clips de val/ son el test: se evaluan una sola vez al final con el
      mejor checkpoint. Esas son las metricas de metrics_<modelo>.json.

Uso (desde la raiz del repo):
    python -m scripts.train_local_cpu --dataset-path data/raw/rwf2000
    python -m scripts.train_local_cpu --model mobilenet_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.models.dataset import VideoDataset, split_by_source
from src.models.metrics import compute_metrics, pr_curve
from src.models.model import MODEL_NAMES, build_model, load_checkpoint, save_checkpoint
from src.models.train import (
    INTERNAL_VAL_FRACTION,
    _run_epoch,
    _stratified_subset,
    set_seed,
)
from src.preprocessing import VideoPreprocessor

logger = logging.getLogger("train_local_cpu")

# Cada cuantos batches se imprime el loss: con batch 4 y ~255 clips son
# ~64 batches por epoca, asi se ve avanzar sin llenar el log.
LOG_EVERY = 10


def parse_args() -> argparse.Namespace:
    """Argumentos con los valores acordados para la corrida en CPU."""
    parser = argparse.ArgumentParser(description="Entrena un modelo en CPU con un subset.")
    # Mismos hiperparametros y subset para todos: asi la tabla comparativa
    # solo cambia el modelo.
    parser.add_argument("--model", type=str, default="mobilenet_lstm", choices=MODEL_NAMES)
    parser.add_argument("--dataset-path", type=str, default="data/raw/rwf2000",
                        help="Carpeta raiz de RWF-2000 (train/ y val/ adentro).")
    parser.add_argument("--train-size", type=int, default=300,
                        help="Clips de train/ (incluye el 15 %% de val interno).")
    parser.add_argument("--test-size", type=int, default=100, help="Clips de val/ para el test.")
    # batch 4: 4 x 16 frames de 224x224 entra holgado en RAM sin GPU.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2,
                        help="Epocas sin mejorar el F1 del val interno antes de cortar.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser.parse_args()


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, epoch: int) -> float:
    """Una epoca de entrenamiento que imprime el loss cada LOG_EVERY batches.

    Hace lo mismo que _run_epoch de train.py en modo train, pero con log por
    batch para seguir la corrida en CPU, que es lenta.

    Returns:
        Loss promedio de la epoca.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    t0 = time.time()
    for step, (tensors, labels) in enumerate(loader, start=1):
        logits = model(tensors)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        if step % LOG_EVERY == 0 or step == len(loader):
            elapsed = time.time() - t0
            eta = elapsed / step * (len(loader) - step)
            logger.info("Epoch %d  batch %d/%d  loss=%.4f  promedio=%.4f  "
                        "%.0fs transcurridos, faltan ~%.0fs",
                        epoch, step, len(loader), loss.item(),
                        total_loss / n_samples, elapsed, eta)
    return total_loss / max(n_samples, 1)


def main() -> int:
    """Arma el subset, entrena con early stopping y evalua el mejor en test."""
    args = parse_args()
    model_name = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # El log va a consola y a archivo, para dejar evidencia de la corrida.
    log_path = output_dir / f"train_{model_name}_local.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    set_seed(args.seed)
    device = torch.device("cpu")  # forzado: esta corrida es la de CPU
    logger.info("Argumentos: %s", vars(args))
    logger.info("torch %s, %d hilos, device=%s", torch.__version__,
                torch.get_num_threads(), device)

    # Igual que train.py: el val interno sale de train/ pero sin augmentation.
    train_dataset = VideoDataset(args.dataset_path, split="train")
    internal_dataset = VideoDataset(args.dataset_path, split="train",
                                    preprocessor=VideoPreprocessor(train=False))
    test_dataset = VideoDataset(args.dataset_path, split="val")

    # Subset balanceado de train/ y split por fuente DENTRO del subset, para
    # que ningun video fuente quede en train y en val interno a la vez.
    subset_idx = _stratified_subset(train_dataset, args.train_size, args.seed).indices
    subset_samples = [train_dataset.samples[i] for i in subset_idx]
    train_pos, val_pos = split_by_source(subset_samples, INTERNAL_VAL_FRACTION, args.seed)
    train_subset = Subset(train_dataset, [subset_idx[i] for i in train_pos])
    val_subset = Subset(internal_dataset, [subset_idx[i] for i in val_pos])
    test_subset = _stratified_subset(test_dataset, args.test_size, args.seed)
    logger.info("Subset: %d train, %d val interno, %d test (de %d y %d disponibles)",
                len(train_subset), len(val_subset), len(test_subset),
                len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    model = build_model(model_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1,
    )

    class_names = train_dataset.classes
    checkpoint_path = output_dir / f"model_{model_name}.pt"
    best_f1, best_epoch, best_pr_curve = -1.0, 0, None
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, y_true, y_pred, y_score = _run_epoch(
            model, val_loader, criterion, None, device, desc=f"Epoch {epoch} val",
        )
        metrics = compute_metrics(y_true, y_pred, class_names=class_names)
        metrics.update({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "epoch_seconds": time.time() - t_epoch})
        history.append(metrics)
        logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
                    "val_f1_macro=%.4f  (%.0fs)", epoch, args.epochs, train_loss,
                    val_loss, metrics["accuracy"], metrics["f1_macro"],
                    metrics["epoch_seconds"])
        scheduler.step(metrics["f1_macro"])

        if metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = metrics["f1_macro"], epoch
            try:
                best_pr_curve = pr_curve(y_true, y_score)
            except ValueError:
                best_pr_curve = None
            save_checkpoint(model, model_name, checkpoint_path)
            logger.info("Nuevo mejor F1 del val interno: %.4f. Checkpoint en %s",
                        best_f1, checkpoint_path)
        if epoch - best_epoch >= args.patience:
            logger.info("Early stopping: %d epocas sin mejorar (mejor epoca %d).",
                        args.patience, best_epoch)
            break

    train_seconds = time.time() - t_start
    (output_dir / f"metrics_{model_name}_internal_val.json").write_text(json.dumps({
        "model": model_name, "device": str(device),
        "evaluated_on": "val interno (15 % del subset de train/, por fuente)",
        "epochs_run": len(history), "best_f1_macro": best_f1, "best_epoch": best_epoch,
        "pr_curve": best_pr_curve, "elapsed_seconds": train_seconds,
        "final": history[-1], "history": history,
    }, indent=2))

    # Test con el MEJOR checkpoint, una sola vez.
    best_model = load_checkpoint(checkpoint_path, device="cpu")
    t_test = time.time()
    test_loss, y_true, y_pred, y_score = _run_epoch(
        best_model, test_loader, criterion, None, device, desc="Test",
    )
    test_seconds = time.time() - t_test
    test_metrics = compute_metrics(y_true, y_pred, class_names=class_names)
    test_metrics["test_loss"] = test_loss
    try:
        test_pr_curve = pr_curve(y_true, y_score)
    except ValueError:
        test_pr_curve = None

    # Mismo formato que metrics_<modelo>.json de train.py, mas los datos del
    # subset, para que nadie lo confunda con una corrida sobre RWF-2000 entero.
    report = {
        "model": model_name,
        "device": str(device),
        "evaluated_on": f"test: {len(y_true)} clips de val/ (subset, no usados al entrenar)",
        "subset": {"train": len(train_subset), "internal_val": len(val_subset),
                   "test": len(test_subset), "seed": args.seed},
        "hyperparams": {"batch_size": args.batch_size, "epochs": args.epochs,
                        "lr": args.lr, "patience": args.patience},
        "n_test": len(y_true),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "internal_val_f1_macro": best_f1,
        "inference_ms": round(1000 * test_seconds / max(len(y_true), 1), 1),
        "pr_curve": test_pr_curve,
        "train_seconds": train_seconds,
        "final": test_metrics,
    }
    metrics_path = output_dir / f"metrics_{model_name}.json"
    metrics_path.write_text(json.dumps(report, indent=2))
    logger.info("TEST  acc=%.4f  f1_macro=%.4f  recall_sospechoso=%.4f  matriz=%s",
                test_metrics["accuracy"], test_metrics["f1_macro"],
                test_metrics["recall_per_class"][class_names[1]],
                test_metrics["confusion_matrix"])
    logger.info("Tiempo real: entrenamiento %.1f min, test %.1f min, total %.1f min",
                train_seconds / 60, test_seconds / 60, (time.time() - t_start) / 60)
    logger.info("Metricas en %s, log en %s", metrics_path, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
