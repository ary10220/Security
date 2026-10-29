"""Script de entrenamiento (fase 3 del plan).

Uso:
    python -m src.models.train --dataset-path data/raw/rwf2000 --epochs 15
    python -m src.models.train --dataset-path data/raw/rwf2000 --smoke-test

Particion de los datos:
    - train/ se divide 85/15 por video fuente (split_by_source): el 85 % se
      entrena y el 15 % es el val interno que elige el mejor checkpoint y
      dispara el early stopping.
    - val/ entero es el test: no se mira durante el entrenamiento. Al final
      se carga el mejor checkpoint y se evalua UNA vez sobre test. Esas son
      las metricas oficiales (artifacts/metrics_<modelo>.json, las del
      informe); las del val interno van a metrics_<modelo>_internal_val.json.

Trabajo en pareja con Grageda (seccion 5 del plan: train.py, metricas,
matriz de confusion y curva PR). Importa la arquitectura de
`src.models.model.build_model`, de modo que si Lorgio implementa el
baseline o la variante ResNet este script los soporta sin cambios.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.dataset import VideoDataset, split_by_source
from src.models.metrics import compute_metrics, pr_curve
from src.models.model import (
    DEFAULT_HYPERPARAMS,
    MODEL_NAMES,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from src.preprocessing import VideoPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Cantidad de videos del smoke test. 10 + 5 alcanza para armar un batch de
# train y uno de val, verificar forward + backward y que todo lo que sale al
# disco (checkpoint + json) sea el correcto.
SMOKE_TRAIN_SIZE = 10
SMOKE_VAL_SIZE = 5

# Proporcion de train/ que se aparta como val interno (decision del equipo).
INTERNAL_VAL_FRACTION = 0.15


def parse_args() -> argparse.Namespace:
    """Define los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena un detector de violencia.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenet_lstm",
                        help="Arquitectura a entrenar.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Carpeta raiz del dataset (train/ val/ clase/ *.mp4).")
    parser.add_argument("--epochs", type=int,
                        default=DEFAULT_HYPERPARAMS["epochs"],
                        help="Numero de epocas.")
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_HYPERPARAMS["batch_size"],
                        help="Tamano de batch del DataLoader.")
    parser.add_argument("--lr", type=float, default=DEFAULT_HYPERPARAMS["lr"],
                        help="Learning rate inicial.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None,
                        help="Dispositivo (por defecto autodetecta GPU).")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Carpeta donde guardar checkpoint y metricas.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad.")
    parser.add_argument("--patience", type=int,
                        default=DEFAULT_HYPERPARAMS["early_stopping_patience"],
                        help="Epocas sin mejorar el F1 del val interno antes de cortar.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Procesos que decodifican videos (0 en Windows, 2 en Colab).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Entrena 1 epoca con 10 train + 5 val para verificar.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Fija semillas de random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_subset(dataset: VideoDataset, total: int, seed: int) -> Subset:
    """Toma `total` muestras balanceando ambas clases.

    En el smoke test interesa que las 2 clases esten presentes: si tomamos
    los primeros 10 indices tal cual, podemos caer con 10 de la misma clase
    y el loss nunca ve la otra.

    Args:
        dataset: dataset del que extraer el subset.
        total: cantidad total de muestras.
        seed: semilla para la mezcla.
    """
    rng = random.Random(seed)
    per_class = max(1, total // 2)

    # Agrupamos indices por etiqueta y mezclamos cada grupo con la seed.
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i, (_, label) in enumerate(dataset.samples):
        by_label.setdefault(label, []).append(i)
    for indices in by_label.values():
        rng.shuffle(indices)

    chosen: list[int] = []
    for label_indices in by_label.values():
        chosen.extend(label_indices[:per_class])
    # Si quedo corto por desbalance en la fuente, completar con lo que haya.
    if len(chosen) < total:
        extra = [i for i in range(len(dataset)) if i not in set(chosen)]
        rng.shuffle(extra)
        chosen.extend(extra[: total - len(chosen)])
    rng.shuffle(chosen)
    return Subset(dataset, chosen[:total])


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, list[int], list[int], list[float]]:
    """Ejecuta una epoca de train (si hay optimizer) o val.

    Returns:
        Tupla `(loss_promedio, y_true, y_pred, y_score)`, donde `y_score` es
        la probabilidad (softmax) de la clase 1, necesaria para la curva PR.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    # torch.enable_grad / no_grad segun fase: en val no gastamos memoria en
    # el grafo computacional.
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for tensors, labels in tqdm(loader, desc=desc, leave=False):
            tensors = tensors.to(device)
            labels = labels.to(device)

            logits = model(tensors)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # loss.item() ya mueve a CPU; multiplicamos por tamano de batch
            # para que la media final no quede sesgada por el ultimo batch.
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
            y_score.extend(logits.softmax(dim=1)[:, 1].detach().cpu().tolist())

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, y_true, y_pred, y_score


def main() -> int:
    """Punto de entrada: arma datasets, modelo y corre el loop de entrenamiento."""
    args = parse_args()
    set_seed(args.seed)

    # Autodetect del device: si no se pide uno, usar GPU si existe.
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construccion del modelo. Los stubs de Lorgio (baseline, resnet_lstm)
    # lanzan NotImplementedError: lo convertimos en un aviso claro para que
    # train.py no reviente mientras Lorgio no cierra su parte.
    try:
        model = build_model(args.model, pretrained=True).to(device)
    except NotImplementedError as exc:
        logger.warning(
            "Modelo '%s' todavia no esta implementado: %s. Saltando entrenamiento.",
            args.model, exc,
        )
        return 0

    logger.info("Modelo '%s' en %s", args.model, device)
    logger.info("Dataset: %s", args.dataset_path)

    # Dos datasets sobre la misma carpeta train/: el val interno tiene que
    # evaluarse sin augmentation, igual que el test. Los samples salen en el
    # mismo orden (carpetas y archivos ordenados), asi que los indices del
    # split valen para los dos.
    train_dataset = VideoDataset(args.dataset_path, split="train")
    internal_dataset = VideoDataset(
        args.dataset_path, split="train",
        preprocessor=VideoPreprocessor(train=False),
    )
    # split="val" crea su preprocessor sin augmentation (train=False).
    test_dataset = VideoDataset(args.dataset_path, split="val")

    train_idx, val_idx = split_by_source(
        train_dataset.samples, INTERNAL_VAL_FRACTION, args.seed,
    )
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(internal_dataset, val_idx)
    test_subset = test_dataset
    logger.info("Split: %d train, %d val interno, %d test",
                len(train_subset), len(val_subset), len(test_subset))

    if args.smoke_test:
        logger.info("Modo smoke test: %d train + %d val, 1 epoca.",
                    SMOKE_TRAIN_SIZE, SMOKE_VAL_SIZE)
        train_subset = _stratified_subset(train_dataset, SMOKE_TRAIN_SIZE, args.seed)
        val_subset = _stratified_subset(internal_dataset, SMOKE_VAL_SIZE, args.seed)
        test_subset = _stratified_subset(test_dataset, SMOKE_VAL_SIZE, args.seed)
        n_epochs = 1
    else:
        n_epochs = args.epochs

    # pin_memory acelera la copia CPU -> GPU; en CPU no aporta nada.
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                     "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    criterion = nn.CrossEntropyLoss()
    # Adam solo sobre params con requires_grad=True: con backbone congelado,
    # mandar los pesos del backbone al optimizer serian megas de memoria
    # desperdiciados en gradientes que son None.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    # mode='max' porque monitoreamos F1 macro (mas alto es mejor).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    class_names = train_dataset.classes
    best_f1 = -1.0
    best_epoch = 0
    best_pr_curve = None
    checkpoint_path = output_dir / f"model_{args.model}.pt"
    metrics_path = output_dir / f"metrics_{args.model}.json"
    internal_metrics_path = output_dir / f"metrics_{args.model}_internal_val.json"

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss, _, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch}/{n_epochs} train",
        )
        val_loss, y_true, y_pred, y_score = _run_epoch(
            model, val_loader, criterion, None, device,
            desc=f"Epoch {epoch}/{n_epochs} val",
        )
        epoch_metrics = compute_metrics(y_true, y_pred, class_names=class_names)
        epoch_metrics.update({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        history.append(epoch_metrics)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1_macro=%.4f",
            epoch, n_epochs, train_loss, val_loss,
            epoch_metrics["accuracy"], epoch_metrics["f1_macro"],
        )
        scheduler.step(epoch_metrics["f1_macro"])

        # Guardar el mejor por F1 macro (ver docstring de compute_metrics).
        if epoch_metrics["f1_macro"] > best_f1:
            best_f1 = epoch_metrics["f1_macro"]
            best_epoch = epoch
            # La curva PR va con el checkpoint que se guarda, porque es el que
            # usara la inferencia y el umbral solo tiene sentido para ese modelo.
            try:
                best_pr_curve = pr_curve(y_true, y_score)
            except ValueError:
                # Val sin ejemplos de la clase 1 (p.ej. smoke test muy chico).
                best_pr_curve = None
            save_checkpoint(model, args.model, checkpoint_path)
            logger.info("Nuevo mejor F1 macro: %.4f. Checkpoint en %s",
                        best_f1, checkpoint_path)

        # Early stopping: si el val interno no mejora en `patience` epocas,
        # seguir solo sobreajusta y gasta GPU.
        if epoch - best_epoch >= args.patience:
            logger.info("Early stopping: %d epocas sin mejorar (mejor epoca %d).",
                        args.patience, best_epoch)
            break

    elapsed = time.time() - t0
    # Metricas del val interno = las de la ultima epoca; mas el mejor F1 visto.
    final_metrics = history[-1] if history else {}
    internal_report = {
        "model": args.model,
        "device": str(device),
        "evaluated_on": "val interno (15 % de train/, por fuente)",
        "epochs_run": len(history),
        "best_f1_macro": best_f1,
        "best_epoch": best_epoch,
        "pr_curve": best_pr_curve,
        "elapsed_seconds": elapsed,
        "final": final_metrics,
        "history": history,
    }
    internal_metrics_path.write_text(json.dumps(internal_report, indent=2))
    logger.info("Metricas del val interno guardadas en %s", internal_metrics_path)

    # Test: se recarga el MEJOR checkpoint (no el de la ultima epoca) y se
    # evalua una sola vez. Es la misma funcion que usa la API para cargarlo.
    best_model = load_checkpoint(checkpoint_path, device=str(device))
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

    # El formato { model, final: {...} } es el que lee el panel de metricas
    # del frontend (frontend/src/lib/api.js). inference_ms incluye decodificar
    # el video: es lo que tarda un clip de punta a punta.
    report = {
        "model": args.model,
        "device": str(device),
        "evaluated_on": "test (val/ completo, no usado al entrenar)",
        "n_test": len(y_true),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "internal_val_f1_macro": best_f1,
        "inference_ms": round(1000 * test_seconds / max(len(y_true), 1), 1),
        "pr_curve": test_pr_curve,
        "train_seconds": elapsed,
        "final": test_metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    logger.info("Metricas de test guardadas en %s", metrics_path)
    logger.info("Resumen: val_interno_f1=%.4f  test_acc=%.4f  test_f1_macro=%.4f  "
                "entrenamiento=%.1fs", best_f1, test_metrics["accuracy"],
                test_metrics["f1_macro"], elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
