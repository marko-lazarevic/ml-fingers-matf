"""Evaluate YOLO instance segmentation model quality."""

# Helperi za evaluaciju modela, prikaz metrika i sample predikcija.
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_best_weights(run_name: str = "cpu_quick_seg") -> Path:
    """Find the most recently modified best.pt for a training run."""
    candidates = [
        path
        for path in SRC_DIR.rglob("weights/best.pt")
        if run_name in path.as_posix()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Nije pronadjen best.pt za run '{run_name}'. "
            "Prvo zavrsi trening ili prosledi putanju do weights fajla."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    with data_yaml.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    split_rel = data.get(split)
    if not split_rel:
        raise ValueError(f"Split '{split}' nije definisan u {data_yaml}")

    split_path = Path(split_rel)
    if split_path.is_absolute():
        split_dir = split_path.resolve()
    else:
        dataset_root_cfg = data.get("path")
        if dataset_root_cfg:
            dataset_root = Path(dataset_root_cfg)
            if not dataset_root.is_absolute():
                # Keep parity with Ultralytics behavior (repo-relative in notebook runs),
                # then fallback to YAML-relative if needed.
                repo_relative_root = (SRC_DIR / dataset_root).resolve()
                dataset_root = repo_relative_root if repo_relative_root.exists() else (data_yaml.parent / dataset_root).resolve()
        else:
            dataset_root = data_yaml.parent.resolve()
        split_dir = (dataset_root / split_path).resolve()

    if not split_dir.exists():
        raise FileNotFoundError(f"Folder za split '{split}' ne postoji: {split_dir}")

    return split_dir


def _collect_metrics(metrics) -> dict[str, float]:
    return {
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
    }


def _print_metrics(summary: dict[str, float], split: str) -> None:
    print(f"\n=== YOLO Seg Evaluation ({split}) ===")
    print(f"Mask mAP50:      {summary['mask_map50']:.4f}")
    print(f"Mask mAP50-95:   {summary['mask_map50_95']:.4f}")
    print(f"Mask Precision:  {summary['mask_precision']:.4f}")
    print(f"Mask Recall:     {summary['mask_recall']:.4f}")
    print(f"Box mAP50:       {summary['box_map50']:.4f}")
    print(f"Box mAP50-95:    {summary['box_map50_95']:.4f}")
    print(f"Box Precision:   {summary['box_precision']:.4f}")
    print(f"Box Recall:      {summary['box_recall']:.4f}")
    print(f"Fitness:         {summary['fitness']:.4f}")


def _show_sample_predictions(
    model: YOLO,
    data_yaml: Path,
    split: str,
    *,
    device: str,
    imgsz: int,
    conf: float,
    show_samples: int,
    seed: int,
) -> None:
    split_dir = _resolve_split_dir(data_yaml, split)
    images = [
        path
        for path in split_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        print(f"Nema slika za vizuelizaciju u {split_dir}")
        return

    rng = random.Random(seed)
    sample_paths = rng.sample(images, k=min(show_samples, len(images)))

    cols = min(3, len(sample_paths))
    rows = (len(sample_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = [axes] if rows == 1 and cols == 1 else list(axes.flat)

    for axis, image_path in zip(axes, sample_paths):
        results = model.predict(
            source=str(image_path),
            device=device,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )
        annotated = results[0].plot()
        axis.imshow(annotated[:, :, ::-1])
        axis.set_title(image_path.name)
        axis.axis("off")

    for axis in axes[len(sample_paths) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def _collect_image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source}")
        return [source]

    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"Nema slika u folderu: {source}")
    return images


def predict_custom_images(
    weights: str | Path,
    source: str | Path,
    *,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> list:
    """Run segmentation inference on custom images or a folder of images."""
    weights = Path(weights).resolve()
    source = Path(source).resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    image_paths = _collect_image_paths(source)
    model = YOLO(str(weights))

    predict_kwargs = {
        "source": [str(path) for path in image_paths],
        "device": device,
        "imgsz": imgsz,
        "conf": conf,
        "verbose": False,
        "save": False,
    }

    results = model.predict(**predict_kwargs)

    if save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Inference on {len(image_paths)} image(s) ===")
    for image_path, result in zip(image_paths, results):
        count = len(result.boxes) if result.boxes is not None else 0
        print(f"{image_path.name}: {count} hand(s) detected")
        if save_dir is not None:
            output_path = save_dir / image_path.name
            result.save(filename=str(output_path))

    if show:
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = [axes] if rows == 1 and cols == 1 else list(axes.flat)

        for axis, image_path, result in zip(axes, image_paths, results):
            annotated = result.plot()
            axis.imshow(annotated[:, :, ::-1])
            count = len(result.boxes) if result.boxes is not None else 0
            axis.set_title(f"{image_path.name} ({count})")
            axis.axis("off")

        for axis in axes[len(results) :]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()

    if save_dir is not None:
        print(f"Saved predictions to: {save_dir}")

    return results


def evaluate_model_quality(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    show_samples: int = 6,
    save_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Validate a trained YOLO segmentation model and optionally show predictions."""
    weights = Path(weights).resolve()
    data_yaml = Path(data_yaml).resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    model = YOLO(str(weights))

    val_kwargs = {
        "data": str(data_yaml),
        "split": split,
        "device": device,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "plots": True,
        "verbose": True,
    }
    if save_dir is not None:
        val_kwargs["project"] = str(Path(save_dir).parent)
        val_kwargs["name"] = Path(save_dir).name
        val_kwargs["exist_ok"] = True

    metrics = model.val(**val_kwargs)
    summary = _collect_metrics(metrics)
    _print_metrics(summary, split)

    if show_samples > 0:
        _show_sample_predictions(
            model,
            data_yaml,
            split,
            device=device,
            imgsz=imgsz,
            conf=conf,
            show_samples=show_samples,
            seed=seed,
        )

    return summary
