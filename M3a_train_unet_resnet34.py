import csv
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# === 1. Configuration ===
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "train" / "images"
MASK_DIR = BASE_DIR / "data" / "train" / "masks"
LOGS_DIR = BASE_DIR / "training_logs"
IMAGE_SIZE = (512, 512)  # width, height
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
THRESHOLD = 0.5
NUM_WORKERS = 4  # Safe default for Windows/PyCharm
EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_COLORS = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 1, (128, 128, 0): 1}
RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image


# === 2. Model: ImageNet-pretrained ResNet34 encoder + U-Net decoder ===
import segmentation_models_pytorch as smp


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.network = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        # Normalize incoming RGB images (0..1) for ImageNet-pretrained ResNet34.
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        normalized = (images - self.image_mean) / self.image_std
        return self.network(normalized)


# === 3. Dataset ===
def list_files(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    files = {}
    for path in directory.iterdir():
        if path.suffix.lower() in EXTENSIONS:
            key = path.stem.lower()
            if key in files:
                raise ValueError(f"Duplicate filename stem: {key}")
            files[key] = path
    return files


def get_pairs(image_dir, mask_dir):
    images, masks = list_files(image_dir), list_files(mask_dir)
    if not images or images.keys() != masks.keys():
        raise ValueError("Image/mask filenames do not match or the directories are empty")
    return [(images[k], masks[k]) for k in sorted(images)]


def load_image(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, RESAMPLING.BILINEAR), dtype=np.float32) / 255.0


def load_mask(path, size):
    with Image.open(path) as mask:
        mask = mask.resize(size, RESAMPLING.NEAREST)

        if mask.mode in ("RGB", "RGBA", "P"):
            rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
            unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)

            # Handle grayscale-like RGB masks.
            if np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(
                rgb[:, :, 1] == rgb[:, :, 2]
            ):
                labels = (rgb[:, :, 0] > 127).astype(np.float32)

            # Handle known corrosion mask colors.
            elif all(tuple(color) in MASK_COLORS for color in unique_colors):
                labels = np.zeros(rgb.shape[:2], dtype=np.float32)
                for color, class_id in MASK_COLORS.items():
                    labels[np.all(rgb == color, axis=-1)] = class_id

            else:
                raise ValueError(
                    f"Unsupported mask colors in {path}: {unique_colors}"
                )

        else:
            gray = np.asarray(mask.convert("L"), dtype=np.uint8)
            unique_values = np.unique(gray)

            if np.all(unique_values <= 3):
                labels = (gray > 0).astype(np.float32)
            else:
                labels = (gray > 127).astype(np.float32)

    return labels


class CorrosionDataset(Dataset):
    def __init__(self, pairs, size, augment=False):
        self.images = np.stack([load_image(im, size) for im, _ in pairs])
        self.masks = np.stack([load_mask(ma, size) for _, ma in pairs])
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy()
        mask = self.masks[index].copy()
        if self.augment:
            if random.random() > 0.5:
                image, mask = np.flip(image, axis=1), np.flip(mask, axis=1)
            if random.random() > 0.5:
                image, mask = np.flip(image, axis=0), np.flip(mask, axis=0)
            image = np.clip((image + random.uniform(-0.2, 0.2) - 0.5) *
                            random.uniform(0.8, 1.2) + 0.5, 0.0, 1.0)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return image, mask


# === 4. Loss and Metrics ===
def bce_dice_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    total = (probability + target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (total + 1)
    return bce + (1 - dice).mean()


def metrics_from_confusion(tn, fp, fn, tp):
    def div(n, d):
        return float(n / d) if d else 0.0
    return {
        "pixel_accuracy": div(tp + tn, tn + fp + fn + tp),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "corrosion_iou": div(tp, tp + fp + fn),
        "corrosion_dice": div(2 * tp, 2 * tp + fp + fn)
    }


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    loss_sum = 0.0
    tn = fp = fn = tp = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * images.size(0)
        predicted = logits.detach().sigmoid() >= THRESHOLD
        actual = masks >= 0.5
        tp += (predicted & actual).sum().item()
        fp += (predicted & ~actual).sum().item()
        fn += (~predicted & actual).sum().item()
        tn += (~predicted & ~actual).sum().item()
    return {"loss": loss_sum / len(loader.dataset), **metrics_from_confusion(tn, fp, fn, tp)}


# === 5. Training ===
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = [
        int(p.name.rsplit("_", 1)[-1])
        for p in LOGS_DIR.glob("3_resnet34_training_*")
        if p.is_dir() and p.name.rsplit("_", 1)[-1].isdigit()
    ]
    run_id = max(existing_ids, default=0) + 1
    run_dir = LOGS_DIR / f"3_resnet34_training_{run_id:03d}"
    run_dir.mkdir()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (run_dir / "training.log").open("w", encoding="utf-8", buffering=1) as logfile:
        sys.stdout, sys.stderr = Tee(original_stdout, logfile), Tee(original_stderr, logfile)
        try:
            train(run_dir, device)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def train(run_dir, device):
    start_time = time.perf_counter()
    pairs = get_pairs(IMAGE_DIR, MASK_DIR)
    train_pairs, validation_pairs = train_test_split(
        pairs, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
    )
    train_set = CorrosionDataset(train_pairs, IMAGE_SIZE, augment=True)
    val_set = CorrosionDataset(validation_pairs, IMAGE_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    model = ResNet34UNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=8, min_lr=5e-6
    )
    parameters = sum(p.numel() for p in model.parameters())
    config = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "task": "Binary segmentation", "framework": "PyTorch",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "image_dir": str(IMAGE_DIR), "mask_dir": str(MASK_DIR),
        "image_size": list(IMAGE_SIZE), "num_classes": 2,
        "architecture": "ResNet34-U-Net", "encoder_name": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE, "validation_split": VALIDATION_SPLIT,
        "random_seed": RANDOM_SEED, "threshold": THRESHOLD,
        "loss": "Binary Crossentropy + Dice Loss", "model_parameters": parameters,
        "training_images": len(train_pairs), "validation_images": len(validation_pairs),
        "training_files": [str(p) for p, _ in train_pairs],
        "validation_files": [str(p) for p, _ in validation_pairs],
        "mask_mapping": "Background=0; Foreground=1"
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("PyTorch:", torch.__version__)
    print("Device:", device, "| GPU:", config["gpu_name"])
    print("Train:", len(train_set), "| Validation:", len(val_set))
    print("Model parameters:", parameters)
    print("Run directory:", run_dir)

    fields = ["epoch", "loss", "pixel_accuracy", "precision", "recall", "corrosion_iou",
              "corrosion_dice", "val_loss", "val_pixel_accuracy", "val_precision", "val_recall",
              "val_corrosion_iou", "val_corrosion_dice", "lr", "epoch_seconds"]
    best_iou, best_epoch, since_best = -1.0, 0, 0
    epoch_times = []
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            training = run_epoch(model, train_loader, optimizer, device, train=True)
            validation = run_epoch(model, val_loader, optimizer, device, train=False)
            scheduler.step(validation["loss"])
            duration = time.perf_counter() - epoch_start
            epoch_times.append(duration)
            row = {"epoch": epoch, **training, **{f"val_{k}": v for k, v in validation.items()},
                   "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": duration}
            writer.writerow(row)
            file.flush()
            print(f"Epoch {epoch:02d}/{EPOCHS} | loss={training['loss']:.4f} "
                  f"| IoU={training['corrosion_iou']:.4f} | val_loss={validation['loss']:.4f} "
                  f"| val_IoU={validation['corrosion_iou']:.4f} | {duration:.1f}s")
            if validation["corrosion_iou"] > best_iou:
                best_iou, best_epoch, since_best = validation["corrosion_iou"], epoch, 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            else:
                since_best += 1
            if since_best >= 25:
                print(f"Early stopping: no improvement in validation IoU for {since_best} epochs")
                break

    config.update({"completed_epochs": len(epoch_times), "best_epoch": best_epoch,
                   "best_validation_iou": best_iou,
                   "average_epoch_seconds": float(np.mean(epoch_times)),
                   "total_execution_seconds": time.perf_counter() - start_time})
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Best validation IoU: {best_iou:.4f} (epoch {best_epoch})")
    print("Saved model:", run_dir / "best_model.pt")


if __name__ == "__main__":
    main()
