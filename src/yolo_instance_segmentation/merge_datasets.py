"""Prepare combined YOLO segmentation dataset config for EgoHands + filtered dataset."""

# Spaja putanje vise datasetova i generise combined_data.yaml za trening.
from __future__ import annotations

from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
EGOHANDS_DIR = SRC_DIR / "data" / "egohands_yolo"
FILTERED_DIR = SRC_DIR / "data" / "yolo_dataset_filtered"
COMBINED_YAML = SRC_DIR / "data" / "combined_data.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_dataset(root: Path, name: str) -> None:
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.exists():
            raise FileNotFoundError(f"{name}: missing {images}")
        if not labels.exists():
            raise FileNotFoundError(f"{name}: missing {labels}")


def prepare_combined_dataset(
    *,
    egohands_dir: Path | None = None,
    filtered_dir: Path | None = None,
    output_yaml: Path | None = None,
) -> Path:
    """Validate both datasets and write combined_data.yaml."""
    egohands_dir = (egohands_dir or EGOHANDS_DIR).resolve()
    filtered_dir = (filtered_dir or FILTERED_DIR).resolve()
    output_yaml = (output_yaml or COMBINED_YAML).resolve()

    _validate_dataset(egohands_dir, "EgoHands")
    _validate_dataset(filtered_dir, "yolo_dataset_filtered")

    data = {
        "train": [
            str((egohands_dir / "images" / "train").resolve()),
            str((filtered_dir / "images" / "train").resolve()),
        ],
        "val": [
            str((egohands_dir / "images" / "val").resolve()),
            str((filtered_dir / "images" / "val").resolve()),
        ],
        "test": str((egohands_dir / "images" / "test").resolve()),
        "names": {0: "hand"},
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    ego_train = _count_images(egohands_dir / "images" / "train")
    ego_val = _count_images(egohands_dir / "images" / "val")
    filt_train = _count_images(filtered_dir / "images" / "train")
    filt_val = _count_images(filtered_dir / "images" / "val")

    print(f"Combined dataset config: {output_yaml}")
    print(f"  EgoHands train/val:   {ego_train} / {ego_val}")
    print(f"  Filtered train/val:   {filt_train} / {filt_val}")
    print(f"  Total train/val:      {ego_train + filt_train} / {ego_val + filt_val}")
    print(f"  Test (EgoHands only): {_count_images(egohands_dir / 'images' / 'test')}")

    return output_yaml


if __name__ == "__main__":
    prepare_combined_dataset()
