"""Tests del split train / val interno por video fuente (src/models/dataset.py)."""

from pathlib import Path

from src.models.dataset import source_id, split_by_source


def _rwf_like(n_sources: int, clips_per_source: int) -> list[tuple[Path, int]]:
    """Arma samples con nombres como RWF-2000: <fuente>_<n>.avi, dos clases."""
    samples = []
    for label, folder in ((1, "Train_Fight"), (0, "Train_NonFight")):
        for s in range(n_sources):
            for n in range(clips_per_source):
                samples.append((Path(folder) / f"{folder}{s}x_{n}.avi", label))
    return samples


def test_source_id_quita_el_numero_de_tramo() -> None:
    assert source_id("data/Train_Fight/_2RYnSFPD_U_0.avi") == "_2RYnSFPD_U"
    assert source_id("abc123_15.avi") == "abc123"
    assert source_id("sin_numero.avi") == "sin_numero"


def test_ninguna_fuente_queda_en_train_y_val() -> None:
    samples = _rwf_like(n_sources=20, clips_per_source=3)
    train, val = split_by_source(samples)

    train_sources = {source_id(samples[i][0]) for i in train}
    val_sources = {source_id(samples[i][0]) for i in val}
    assert train_sources.isdisjoint(val_sources)


def test_cubre_todo_sin_repetir() -> None:
    samples = _rwf_like(n_sources=20, clips_per_source=3)
    train, val = split_by_source(samples)

    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(len(samples)))


def test_estratificado_por_clase() -> None:
    samples = _rwf_like(n_sources=20, clips_per_source=3)
    _, val = split_by_source(samples, val_fraction=0.15)

    # 15 % de 20 fuentes = 3 fuentes por clase = 9 clips por clase.
    val_labels = [samples[i][1] for i in val]
    assert val_labels.count(1) == 9
    assert val_labels.count(0) == 9


def test_misma_seed_mismo_split() -> None:
    samples = _rwf_like(n_sources=20, clips_per_source=3)
    assert split_by_source(samples, seed=42) == split_by_source(samples, seed=42)
    assert split_by_source(samples, seed=42) != split_by_source(samples, seed=7)


def test_una_sola_fuente_por_clase_divide_por_clip() -> None:
    # Como el mini dataset: fight_01..06 son todos de la "fuente" fight.
    samples = [(Path(f"Fight/fight_{n:02d}.mp4"), 1) for n in range(1, 7)]
    samples += [(Path(f"NonFight/non_{n:02d}.mp4"), 0) for n in range(1, 7)]
    train, val = split_by_source(samples)

    assert {samples[i][1] for i in val} == {0, 1}
    assert {samples[i][1] for i in train} == {0, 1}
