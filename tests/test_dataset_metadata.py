"""Tests de src/preprocessing/dataset_metadata.py (metadatos para el EDA)."""

import shutil
from pathlib import Path

import pytest

from src.preprocessing.dataset_metadata import (
    label_from_path,
    read_video_info,
    scan_dataset,
    split_from_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "ruta, esperada",
    [
        ("Violence/V_1.mp4", "violencia"),
        ("NonViolence/NV_1.mp4", "no_violencia"),
        ("train/Train_Fight/a.avi", "violencia"),
        ("val/Val_NonFight/b.avi", "no_violencia"),
    ],
)
def test_label_from_path(ruta: str, esperada: str) -> None:
    # "NonViolence" contiene "violence": el orden de las reglas importa.
    assert label_from_path(Path(ruta)) == esperada


def test_label_from_path_sin_clase() -> None:
    with pytest.raises(ValueError):
        label_from_path(Path("otros/video.mp4"))


def test_split_from_path() -> None:
    assert split_from_path(Path("train/Train_Fight/a.avi")) == "train"
    assert split_from_path(Path("val/Val_Fight/a.avi")) == "val"
    assert split_from_path(Path("Violence/V_1.mp4")) == ""


def test_read_video_info_fixture() -> None:
    info = read_video_info(FIXTURES / "sample_video.mp4")
    assert info["legible"]
    assert (info["ancho"], info["alto"]) == (320, 240)
    assert info["frames"] == 90
    assert info["segundos"] == pytest.approx(info["frames"] / info["fps"])


def test_read_video_info_corrupto() -> None:
    assert not read_video_info(FIXTURES / "video_corrupto.mp4")["legible"]


def test_scan_dataset(tmp_path: Path) -> None:
    # Se arma un mini dataset con la misma estructura que Real Life.
    for clase in ("Violence", "NonViolence"):
        (tmp_path / clase).mkdir()
        shutil.copy(FIXTURES / "sample_video.mp4", tmp_path / clase / "x.mp4")
    (tmp_path / "Violence" / "notas.txt").write_text("no es un video")

    df = scan_dataset(tmp_path, "real_life")

    assert len(df) == 2
    assert set(df["clase"]) == {"violencia", "no_violencia"}
    assert (df["dataset"] == "real_life").all()
    assert df["legible"].all()
