"""Recorta clips etiquetados y comprueba que no haya fuga por video fuente.

Uso:
    python -m scripts.prepare_risk_data --manifest data/risk_labels.csv \
        --output data/processed/risk_modes

El CSV usa: file,source_id,mode,label,start_sec,end_sec,split,rights,notes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2

REQUIRED = {"file", "source_id", "mode", "label", "start_sec", "end_sec", "split", "rights", "notes"}
MODES = {"interior", "exterior"}
LABELS = {"Normal", "Riesgo"}
SPLITS = {"train", "val"}


def read_manifest(path: Path) -> list[dict]:
    """Valida etiquetas, rutas e intervalos antes de crear ningún clip.

    Args:
        path: CSV con una fila por intervalo revisado.

    Returns:
        Filas con rutas absolutas y tiempos numéricos.
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Faltan columnas: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("El manifiesto no contiene clips.")
    for number, row in enumerate(rows, 2):
        if row["mode"] not in MODES or row["label"] not in LABELS or row["split"] not in SPLITS:
            raise ValueError(f"Fila {number}: mode, label o split inválido.")
        if not row["source_id"].strip() or not row["rights"].strip():
            raise ValueError(f"Fila {number}: faltan source_id o rights.")
        source = (path.parent / row["file"]).resolve()
        if not source.is_file():
            raise ValueError(f"Fila {number}: no existe {source}.")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not 0 <= start < end or end - start > 30:
            raise ValueError(f"Fila {number}: intervalo inválido o mayor a 30 s.")
        row["source_path"] = source
        row["start"] = start
        row["end"] = end
    return rows


def validate_splits(rows: list[dict]) -> None:
    """Exige fuentes separadas y ejemplos de ambas clases en cada partición.

    Args:
        rows: filas ya verificadas del manifiesto.
    """
    for mode in {row["mode"] for row in rows}:
        subset = [row for row in rows if row["mode"] == mode]
        train = {row["source_id"] for row in subset if row["split"] == "train"}
        test = {row["source_id"] for row in subset if row["split"] == "val"}
        if train & test:
            raise ValueError(f"{mode}: fuentes repetidas entre train y test: {train & test}")
        for label in LABELS:
            sources = {row["source_id"] for row in subset
                       if row["split"] == "train" and row["label"] == label}
            test_count = sum(row["split"] == "val" and row["label"] == label
                             for row in subset)
            if len(sources) < 2 or test_count < 1:
                raise ValueError(f"{mode}/{label}: se requieren 2 fuentes train y 1 clip test.")


def write_clip(source: Path, target: Path, start: float, end: float) -> None:
    """Reencodifica el intervalo relevante antes del entrenamiento.

    Args:
        source: video original sin recortar.
        target: clip MP4 de salida.
        start: inicio revisado en segundos.
        end: fin revisado en segundos.
    """
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or end > frame_count / fps + 0.1:
            raise ValueError(f"Video ilegible o intervalo fuera de duración: {source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear {target}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        written = 0
        while capture.get(cv2.CAP_PROP_POS_FRAMES) < end * fps:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        if written < 2:
            raise ValueError(f"Intervalo sin fotogramas suficientes: {source}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"Clip vacío: {target}")


def main() -> int:
    """Valida el CSV y materializa únicamente los intervalos etiquetados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    validate_splits(rows)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("La carpeta de salida debe estar vacía para evitar clips anteriores.")
    manifest = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        source_key = (row["mode"], row["source_id"])
        index = counters[source_key]
        counters[source_key] += 1
        source_hash = hashlib.sha256(row["source_id"].encode("utf-8")).hexdigest()[:16]
        target = args.output / row["mode"] / row["split"] / row["label"] / f"{source_hash}_{index}.mp4"
        write_clip(row["source_path"], target, row["start"], row["end"])
        manifest.append({"clip": str(target.relative_to(args.output)),
                         "source_id": row["source_id"], "label": row["label"],
                         "start_sec": row["start"], "end_sec": row["end"],
                         "rights": row["rights"], "notes": row["notes"]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Preparados {len(manifest)} clips en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
