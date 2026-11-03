"""Pruebas de scripts/analisis_resolucion.py con predicciones sinteticas."""

from __future__ import annotations

import pytest

from scripts.analisis_resolucion import analizar


def _filas(y_true: list[int], y_pred: list[int], altos: list[int]) -> list[dict]:
    """Arma filas con el formato que devuelve `leer_predicciones`."""
    return [
        {"video": f"v{i}.avi", "y_true": t, "y_pred": p, "ancho": 2 * a, "alto": a}
        for i, (t, p, a) in enumerate(zip(y_true, y_pred, altos))
    ]


def test_regla_y_modelo_se_miden_sobre_los_mismos_clips() -> None:
    """La regla acierta cuando el alto separa las clases; el modelo se mide aparte."""
    y_true = [1, 1, 0, 0]
    altos = [720, 720, 240, 240]
    resultado = analizar({"m": _filas(y_true, [1, 0, 0, 0], altos)}, alto_umbral=251)
    assert resultado["regla_exactitud"] == 1.0
    assert resultado["modelos"]["m"]["exactitud"] == 0.75
    assert resultado["modelos"]["m"]["acuerdo_con_regla"] == 0.75


def test_linea_mayoritaria_por_resolucion() -> None:
    """En un grupo con 3 de 4 NonFight, contestar siempre NonFight acierta 0,75."""
    y_true = [0, 0, 0, 1]
    resultado = analizar({"m": _filas(y_true, [0, 0, 0, 0], [240] * 4)})
    grupo = resultado["por_resolucion"][0]
    assert grupo["n"] == 4
    assert grupo["linea_mayoritaria"] == 0.75


def test_rechaza_modelos_con_clips_distintos() -> None:
    """Comparar clip por clip exige que todos los modelos tengan los mismos videos."""
    a = _filas([1, 0], [1, 0], [720, 240])
    b = _filas([1, 0], [1, 0], [720, 240])
    b[1]["video"] = "otro.avi"
    with pytest.raises(ValueError):
        analizar({"a": a, "b": b})
