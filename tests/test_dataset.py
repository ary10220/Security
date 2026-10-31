"""Tests de los nombres de clase de VideoDataset (src/models/dataset.py)."""

from pathlib import Path

import pytest

from src.models.dataset import VideoDataset, class_display_name


def _make_split(root: Path, split: str, folders: tuple[str, str]) -> None:
    """Crea carpetas de clase con un .avi vacio: VideoDataset solo lista archivos."""
    for folder in folders:
        class_dir = root / split / folder
        class_dir.mkdir(parents=True)
        (class_dir / "abc_0.avi").touch()


@pytest.mark.parametrize("folder, expected", [
    ("Train_Fight", "Fight"),
    ("Val_NonFight", "NonFight"),
    ("test_Fight", "Fight"),
    ("NonViolence", "NonViolence"),
    ("Violence", "Violence"),
])
def test_class_display_name_quita_prefijo_de_split(folder: str, expected: str) -> None:
    assert class_display_name(folder) == expected


def test_classes_de_rwf_sin_prefijo(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", ("Train_Fight", "Train_NonFight"))
    _make_split(tmp_path, "val", ("Val_Fight", "Val_NonFight"))

    train = VideoDataset(tmp_path, split="train")
    val = VideoDataset(tmp_path, split="val")

    # Orden por label: 0 = normal, 1 = sospechoso, igual en los dos splits.
    assert train.classes == ["NonFight", "Fight"]
    assert val.classes == train.classes


def test_labels_siguen_igual_tras_quitar_prefijo(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", ("Train_Fight", "Train_NonFight"))

    dataset = VideoDataset(tmp_path, split="train")
    labels = {path.parent.name: label for path, label in dataset.samples}

    assert labels == {"Train_Fight": 1, "Train_NonFight": 0}
