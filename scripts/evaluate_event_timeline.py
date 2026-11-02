"""Evalúa la decisión temporal de la API en clips de prueba no entrenados.

Cada fila positiva del manifiesto representa un evento visible en el clip
recortado. El recall resultante aplica a esos eventos anotados, no a videos
largos completos ni a otras cámaras.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api import model_loader
from api.main import DEFAULT_THRESHOLD, TimelineSegment, _score_segments, _segment_bounds, _verdict, _video_duration
from api.modes import MODES, mode_status
from src.models.metrics import compute_metrics


def evaluate(root: Path, mode: str) -> dict:
    """Mide recall de eventos y falsas alertas con las ventanas reales de API.

    Args:
        root: carpeta creada por `prepare_risk_data`, con `manifest.json`.
        mode: `interior` o `exterior`.

    Returns:
        Reporte de prueba temporal y casos errados.
    """
    if mode not in ("interior", "exterior"):
        raise ValueError("Este protocolo requiere interior o exterior.")
    status = mode_status(mode)
    if not status["available"]:
        raise ValueError(f"Modelo no disponible: {status['detail']}")
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in rows if Path(row["clip"]).parts[:2] == (mode, "val")]
    if {row["label"] for row in rows} != {"Normal", "Riesgo"}:
        raise ValueError("La prueba necesita eventos y negativos etiquetados.")
    loaded = model_loader.load_model(MODES[mode].artifact, expected_mode=mode)
    actual, predicted, errors = [], [], []
    for row in rows:
        path = root / row["clip"]
        duration = _video_duration(path)
        bounds = _segment_bounds(0, duration)
        scores = _score_segments(path, bounds, loaded)
        timeline = [TimelineSegment(start=start, end=end, score=score)
                    for (start, end), score in zip(bounds, scores)]
        verdict, _ = _verdict(timeline, DEFAULT_THRESHOLD)
        truth = int(row["label"] == "Riesgo")
        guess = int(verdict == "suspicious")
        actual.append(truth)
        predicted.append(guess)
        if truth != guess:
            errors.append({"clip": row["clip"], "source_id": row["source_id"],
                           "type": "false_alert" if truth == 0 else "missed_event",
                           "peak_score": round(max(scores), 4), "notes": row["notes"]})
    measured = compute_metrics(actual, predicted, class_names=["Normal", "Riesgo"])
    return {"mode": mode, "protocol": "un evento anotado por clip positivo recortado",
            "n_test_clips": len(rows),
            "n_events": sum(actual),
            "event_recall": measured["recall_per_class"]["Riesgo"],
            "false_alerts": measured["confusion_matrix"][0][1],
            "confusion_matrix": measured["confusion_matrix"],
            "errors": errors}


def main() -> int:
    """Calcula y guarda una evaluación temporal separada de la de train.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("interior", "exterior"), required=True)
    args = parser.parse_args()
    report = evaluate(args.dataset_root, args.mode)
    output = model_loader.ARTIFACTS_DIR / f"timeline_eval_{args.mode}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluación temporal guardada en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
