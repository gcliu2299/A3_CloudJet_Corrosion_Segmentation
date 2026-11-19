import argparse
import csv
import json
import importlib
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

train_module = importlib.import_module("M3a_train_unet_resnet34")
BASE_DIR = train_module.BASE_DIR
LOGS_DIR = train_module.LOGS_DIR
ResNet34UNet = train_module.ResNet34UNet
get_pairs = train_module.get_pairs
load_image = train_module.load_image
load_mask = train_module.load_mask

# === 1. Configuration ===
SELECTED_RUN = None  # e.g. "3_resnet34_training_002"; None selects latest completed run
THRESHOLD = 0.60
THRESHOLDS = np.round(np.arange(0.30, 0.901, 0.05), 2)
VISUALIZE = True
NUM_VISUALIZATIONS = 10


# === 2. Model Selection ===
def select_run(name=None):
    if name:
        folder = Path(name)
        folder = folder if folder.is_absolute() else LOGS_DIR / folder
        if not (folder / "best_model.pt").is_file():
            raise FileNotFoundError(folder / "best_model.pt")
        return folder
    if not LOGS_DIR.is_dir():
        raise FileNotFoundError(LOGS_DIR)
    folders = sorted(
        (p for p in LOGS_DIR.iterdir() if p.is_dir() and p.name.startswith("3_resnet34_training_")),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
        reverse=True
    )
    for folder in folders:
        if (folder / "best_model.pt").is_file() and (folder / "config.json").is_file():
            return folder
    raise FileNotFoundError("No completed PyTorch training run found")


def validation_pairs(config):
    image_dir, mask_dir = Path(config["image_dir"]), Path(config["mask_dir"])
    lookup = {image.stem.lower(): (image, mask) for image, mask in get_pairs(image_dir, mask_dir)}
    pairs = []
    for saved in config["validation_files"]:
        key = Path(saved).stem.lower()
        if key not in lookup:
            raise FileNotFoundError(f"Validation pair missing: {saved}")
        pairs.append(lookup[key])
    return pairs


# === 3. Metrics ===
def safe_ratio(a, b):
    return float(a / b) if b else 0.0


def evaluate_threshold(predictions, threshold):
    tn = fp = fn = tp = 0
    areas = []
    for name, truth, probability in predictions:
        predicted = probability >= threshold
        actual = truth >= 0.5
        tn += int((~predicted & ~actual).sum())
        fp += int((predicted & ~actual).sum())
        fn += int((~predicted & actual).sum())
        tp += int((predicted & actual).sum())
        areas.append((name, int(actual.sum()), int(predicted.sum()), int(actual.size)))
    confusion = np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)
    abs_fraction = [abs(p - t) / total for _, t, p, total in areas]
    bias_fraction = [(p - t) / total for _, t, p, total in areas]
    metrics = {
        "threshold": float(threshold),
        "pixel_accuracy": safe_ratio(tp + tn, tn + fp + fn + tp),
        "corrosion_iou": safe_ratio(tp, tp + fp + fn),
        "corrosion_dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "corrosion_precision": safe_ratio(tp, tp + fp),
        "corrosion_recall": safe_ratio(tp, tp + fn),
        "background_iou": safe_ratio(tn, tn + fp + fn),
        "mean_iou": (safe_ratio(tp, tp + fp + fn) + safe_ratio(tn, tn + fp + fn)) / 2,
        "true_corrosion_pixels": tp + fn,
        "predicted_corrosion_pixels": tp + fp,
        "true_corrosion_fraction": safe_ratio(tp + fn, confusion.sum()),
        "predicted_corrosion_fraction": safe_ratio(tp + fp, confusion.sum()),
        "mean_absolute_area_error_pixels": float(np.mean([abs(p - t) for _, t, p, _ in areas])),
        "mean_absolute_area_error_fraction": float(np.mean(abs_fraction)),
        "mean_signed_area_error_fraction": float(np.mean(bias_fraction))
    }
    return metrics, confusion, areas


# === 4. GPU Inference ===
def collect_predictions(model, pairs, size, device):
    predictions, latencies = [], []
    model.eval()
    with torch.inference_mode():
        warmup = torch.zeros((1, 3, size[1], size[0]), dtype=torch.float32, device=device)
        for _ in range(10):
            model(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for index, (image_path, mask_path) in enumerate(pairs, 1):
            image = load_image(image_path, size)
            truth = load_mask(mask_path, size).astype(bool)
            tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            probability = model(tensor).sigmoid()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            probability = probability[0, 0].cpu().numpy()
            latencies.append(elapsed)
            predictions.append((image_path.name, truth, probability))
            if index % 20 == 0 or index == len(pairs):
                print(f"Evaluated {index}/{len(pairs)} images")
    return predictions, latencies


# === 5. Visualizations ===
def save_visualizations(folder, predictions, pairs, threshold, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = folder / "visualizations"
    output_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(predictions), min(count, len(predictions)), replace=False)
    for index in indices:
        name, truth, probability = predictions[index]
        predicted = probability >= threshold
        image = load_image(pairs[index][0], (truth.shape[1], truth.shape[0]))
        error = np.zeros((*truth.shape, 3), dtype=np.uint8)
        error[predicted & ~truth] = [255, 0, 0]  # False positive
        error[~predicted & truth] = [0, 0, 255]  # False negative
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, data, title in zip(
            axes, (image, truth, predicted, error),
            ("Original", "Ground Truth", "Prediction", "Error Map")
        ):
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None,
                      vmin=0 if data.ndim == 2 else None,
                      vmax=1 if data.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(name).stem}.png", dpi=150)
        plt.close(fig)
    print("Visualizations saved to:", output_dir)


# === 6. Save Results ===
def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=SELECTED_RUN)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args, _ = parser.parse_known_args()
    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    folder = select_run(args.run)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    size = tuple(config["image_size"])
    pairs = validation_pairs(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("architecture") != "ResNet34-U-Net":
        raise ValueError("Selected run is not a ResNet34-U-Net experiment")
    model = ResNet34UNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location=device, weights_only=True))
    print("PyTorch:", torch.__version__, "| Device:", device)
    print("Model:", folder / "best_model.pt")
    print("Validation images:", len(pairs))
    predictions, latencies = collect_predictions(model, pairs, size, device)

    metrics, confusion, areas = evaluate_threshold(predictions, args.threshold)
    mean_latency = float(np.mean(latencies))
    metrics.update({
        "inference_mean_ms": mean_latency,
        "inference_median_ms": float(np.median(latencies)),
        "inference_p95_ms": float(np.percentile(latencies, 95)),
        "inference_fps_estimate": safe_ratio(1000, mean_latency),
        "model_path": str(folder / "best_model.pt"),
        "evaluation_set": "Validation", "evaluation_images": len(pairs),
        "area_unit": "pixels at resized resolution (not mm2)",
        "torch_version": torch.__version__
    })
    (folder / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (folder / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["True / Predicted", "Background", "Corrosion"])
        writer.writerow(["Background", *confusion[0]])
        writer.writerow(["Corrosion", *confusion[1]])
    save_csv(folder / "corrosion_area_by_image.csv", [
        {"image": name, "true_pixels": t, "predicted_pixels": p,
         "absolute_error_pixels": abs(p - t), "true_fraction": t / total,
         "predicted_fraction": p / total}
        for name, t, p, total in areas
    ], ["image", "true_pixels", "predicted_pixels", "absolute_error_pixels", "true_fraction", "predicted_fraction"])

    sweep = [evaluate_threshold(predictions, float(t))[0] for t in THRESHOLDS]
    fields = ["threshold", "corrosion_iou", "corrosion_dice", "corrosion_precision",
              "corrosion_recall", "pixel_accuracy", "mean_absolute_area_error_fraction",
              "mean_signed_area_error_fraction"]
    save_csv(folder / "threshold_sweep.csv", sweep, list(sweep[0].keys()))
    best_iou = max(sweep, key=lambda row: row["corrosion_iou"])
    best_area = min(sweep, key=lambda row: row["mean_absolute_area_error_fraction"])
    if VISUALIZE:
        save_visualizations(folder, predictions, pairs, args.threshold, NUM_VISUALIZATIONS)

    print(f"Baseline threshold: {args.threshold:.2f} | IoU: {metrics['corrosion_iou']:.4f} "
          f"| Dice: {metrics['corrosion_dice']:.4f} | Area MAE: "
          f"{100 * metrics['mean_absolute_area_error_fraction']:.2f} pp")
    print(f"Best IoU threshold: {best_iou['threshold']:.2f} "
          f"| IoU: {best_iou['corrosion_iou']:.4f}")
    print(f"Best Area threshold: {best_area['threshold']:.2f} "
          f"| MAE: {100 * best_area['mean_absolute_area_error_fraction']:.2f} pp")
    print(f"Inference latency: {mean_latency:.2f} ms/image")
    print("Saved results to:", folder)


if __name__ == "__main__":
    main()
