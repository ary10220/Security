"""Tests de las ventanas de clips de scripts/build_ashen_manifest.py."""

import pytest

from scripts.build_ashen_manifest import MAX_RISK_CLIPS, normal_windows, risk_windows


def test_evento_corto_da_un_solo_clip_con_todo_el_evento() -> None:
    # 90 a 180 fotogramas a 30 fps: evento de 3 s.
    assert risk_windows([90, 180, -1, -1], fps=30.0, duration=60.0) == [(3.0, 6.0)]


def test_evento_de_menos_de_dos_segundos_se_descarta() -> None:
    assert risk_windows([0, 30, -1, -1], fps=30.0, duration=60.0) == []


def test_convierte_fotogramas_con_el_fps_del_video() -> None:
    # El mismo intervalo en fotogramas dura distinto a 25 fps.
    assert risk_windows([0, 250, -1, -1], fps=25.0, duration=60.0) == [(0.0, 5.0), (5.0, 10.0)]


def test_usa_el_segundo_intervalo_si_existe() -> None:
    windows = risk_windows([0, 150, 300, 450], fps=30.0, duration=60.0)
    assert windows == [(0.0, 5.0), (10.0, 15.0)]


def test_evento_largo_queda_limitado_y_repartido() -> None:
    windows = risk_windows([0, 3000, -1, -1], fps=30.0, duration=200.0)
    assert len(windows) == MAX_RISK_CLIPS
    # Repartidas: la ultima no es la sexta ventana desde el inicio.
    assert windows[0] == (0.0, 5.0)
    assert windows[-1][0] > 5 * (MAX_RISK_CLIPS - 1)


def test_el_evento_no_pasa_del_final_del_video() -> None:
    windows = risk_windows([0, 900, -1, -1], fps=30.0, duration=12.0)
    assert all(end <= 12.0 - 0.2 for _, end in windows)


@pytest.mark.parametrize("duration, expected", [(16.0, 3), (12.3, 2), (5.5, 0)])
def test_normal_reparte_hasta_tres_ventanas(duration: float, expected: int) -> None:
    windows = normal_windows(duration)
    assert len(windows) == expected
    assert all(end - start == pytest.approx(5.0) and end <= duration for start, end in windows)
