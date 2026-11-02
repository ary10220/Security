"""Tests de la seleccion de clips y nombres de scripts/eval_test_por_clip.py."""

from pathlib import Path

from scripts.eval_test_por_clip import artifact_name, label_map_for, select_test_indices
from src.models.dataset import VideoDataset
from src.models.train import _stratified_subset


def _make_val(root: Path, folders: tuple[str, str], per_class: int) -> VideoDataset:
    """Crea val/ con videos vacios: VideoDataset solo lista archivos."""
    for folder in folders:
        class_dir = root / "val" / folder
        class_dir.mkdir(parents=True)
        for i in range(per_class):
            (class_dir / f"src{i}_{i}.avi").touch()
    return VideoDataset(root, split="val")


def test_test_size_cero_usa_val_completo(tmp_path: Path) -> None:
    dataset = _make_val(tmp_path, ("Val_Fight", "Val_NonFight"), per_class=6)
    assert select_test_indices(dataset, test_size=0, seed=42) == list(range(12))


def test_test_size_positivo_conserva_el_subset_anterior(tmp_path: Path) -> None:
    # Comportamiento previo del script: el mismo subset estratificado que
    # train_local_cpu.py. Cambiarlo romperia la comparacion con la corrida CPU.
    dataset = _make_val(tmp_path, ("Val_Fight", "Val_NonFight"), per_class=6)
    expected = list(_stratified_subset(dataset, 4, 42).indices)
    assert select_test_indices(dataset, test_size=4, seed=42) == expected


def test_artifact_name_sigue_la_regla_de_train() -> None:
    assert artifact_name("mobilenet_lstm", "peleas") == "mobilenet_lstm"
    assert artifact_name("resnet_lstm", "peleas") == "resnet_lstm"
    assert artifact_name("mobilenet_lstm", "interior") == "interior"


def test_label_map_por_modo() -> None:
    assert label_map_for("peleas") is None
    assert label_map_for("exterior") == {"Normal": 0, "Riesgo": 1}
