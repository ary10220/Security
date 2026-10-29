"""Tests de las metricas de evaluacion (sin modelo ni videos)."""

import pytest

from src.models.metrics import compute_metrics, pr_curve


def test_compute_metrics_perfecto():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], class_names=["a", "b"])
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_pr_curve_separable_da_ap_uno():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    r = pr_curve(y_true, y_score)
    assert r["average_precision"] == pytest.approx(1.0)
    assert r["best_f1"] == pytest.approx(1.0)
    # Cualquier umbral entre 0.3 y 0.7 separa perfecto; el mejor es el primero
    # que alcanza F1 = 1, es decir, el score mas bajo de la clase positiva.
    assert r["best_f1_threshold"] == pytest.approx(0.7)


def test_pr_curve_listas_paralelas_y_rangos():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.4, 0.6, 0.5, 0.45, 0.9, 0.2, 0.55, 0.65]
    r = pr_curve(y_true, y_score)
    n = len(r["thresholds"])
    assert len(r["precision"]) == len(r["recall"]) == n > 0
    assert all(0.0 <= v <= 1.0 for v in r["precision"] + r["recall"])
    assert 0.0 <= r["average_precision"] <= 1.0
    assert r["best_f1_threshold"] in r["thresholds"]


def test_pr_curve_recorta_puntos_sin_perder_el_mejor_umbral():
    n = 1000
    y_true = [i % 2 for i in range(n)]
    y_score = [(i * 7919 % n) / n for i in range(n)]
    completa = pr_curve(y_true, y_score, max_points=None)
    corta = pr_curve(y_true, y_score, max_points=50)
    assert len(corta["thresholds"]) <= 50
    assert len(completa["thresholds"]) > 50
    assert corta["best_f1_threshold"] == completa["best_f1_threshold"]
    assert corta["average_precision"] == completa["average_precision"]


def test_pr_curve_es_serializable_a_json():
    import json
    r = pr_curve([0, 1, 1, 0], [0.2, 0.8, 0.6, 0.4])
    assert json.loads(json.dumps(r)) == r


def test_pr_curve_sin_positivos_falla_claro():
    with pytest.raises(ValueError, match="clase 1"):
        pr_curve([0, 0, 0], [0.1, 0.2, 0.3])


def test_pr_curve_formas_distintas_falla():
    with pytest.raises(ValueError, match="misma forma"):
        pr_curve([0, 1], [0.1, 0.2, 0.3])
