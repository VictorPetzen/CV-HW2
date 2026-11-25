"""
Homework 2 - Image Classification (Computer Vision, Fall 2026)

Shared code used by all the other scripts:
  - constants and paths
  - random seeds / deterministic settings
  - Oxford-IIIT Pet dataset loading and the stratified train/val split
  - MobileNetV3-Small with a new 37-class output layer
  - small helpers (data loaders, prediction loop, parameter count, saving json)
"""

import os
import csv
import json
import random
import platform

import numpy as np
import torch
import torch.nn as nn
import torchvision
import sklearn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Settings (the same for all three approaches)
# ------------------------------------------------------------
SEED = 42
NUM_CLASSES = 37
VAL_FRACTION = 0.2      # 20% of the official trainval split goes to validation
BATCH_SIZE = 32
NUM_WORKERS = 4         # set to 0 if the data loader gives problems on Windows

DATA_DIR = "data"               # the dataset is downloaded here (not in the repo)
SPLIT_DIR = "splits"            # train/val split files (in the repo)
RESULTS_DIR = "results"         # metrics, predictions, plots (in the repo)
CHECKPOINT_DIR = "checkpoints"  # selected model weights

# ImageNet-1K weights of MobileNetV3-Small (= IMAGENET1K_V1)
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Seeds
# ------------------------------------------------------------
def set_seed(seed=SEED):
    """Fix the seeds of Python, NumPy and PyTorch and ask for deterministic ops."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed by deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: a few CUDA kernels (e.g. the backward of adaptive average
    # pooling) have no deterministic version, PyTorch only prints a warning for them
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Every data loader worker gets a seed derived from the main one."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
# Evaluation: exactly the preprocessing of the pre-trained weights
# (resize to 256, center crop 224, ImageNet mean/std normalization)
EVAL_TRANSFORM = WEIGHTS.transforms()

# Training: same normalization, plus light augmentation (training set only)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=EVAL_TRANSFORM.mean, std=EVAL_TRANSFORM.std),
])


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
def load_pets(split, transform):
    """Official Oxford-IIIT Pet split ('trainval' or 'test'), 37 breed labels."""
    return datasets.OxfordIIITPet(root=DATA_DIR, split=split, target_types="category",
                                  transform=transform, download=True)


def get_image_names(ds):
    # torchvision keeps the image paths in a private list, we only read it
    return [os.path.basename(p) for p in ds._images]


def get_labels(ds):
    # same for the labels (reading them this way avoids opening every image)
    return list(ds._labels)


def make_or_load_split(trainval):
    """
    Stratified split of the official trainval set into train and val.
    The split is saved in splits/ the first time, and then always loaded from
    there, so all approaches (and all runs) use exactly the same images.
    """
    train_file = os.path.join(SPLIT_DIR, "train.csv")
    val_file = os.path.join(SPLIT_DIR, "val.csv")
    names = get_image_names(trainval)

    if not os.path.exists(train_file):
        os.makedirs(SPLIT_DIR, exist_ok=True)
        labels = get_labels(trainval)
        idx = np.arange(len(labels))
        train_idx, val_idx = train_test_split(idx, test_size=VAL_FRACTION,
                                              stratify=labels, random_state=SEED)
        for path, ids in [(train_file, train_idx), (val_file, val_idx)]:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "image", "label"])
                for i in sorted(ids):
                    w.writerow([i, names[i], labels[i]])
        print(f"Created new split: {len(train_idx)} train / {len(val_idx)} val")

    def read(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        ids = [int(r["index"]) for r in rows]
        # check that the file still matches the dataset on disk
        for r in rows:
            assert names[int(r["index"])] == r["image"], "split file does not match the dataset"
        return ids

    return read(train_file), read(val_file)


def get_datasets():
    """
    Returns a dict with:
      train      - training images with augmentation
      train_eval - the same training images without augmentation (for train accuracy / features)
      val        - validation images
      test       - official test images (used only for the final evaluation)
    and the list of class names.
    """
    trainval_aug = load_pets("trainval", TRAIN_TRANSFORM)
    trainval_plain = load_pets("trainval", EVAL_TRANSFORM)
    test = load_pets("test", EVAL_TRANSFORM)
    train_idx, val_idx = make_or_load_split(trainval_plain)

    sets = {
        "train": Subset(trainval_aug, train_idx),
        "train_eval": Subset(trainval_plain, train_idx),
        "val": Subset(trainval_plain, val_idx),
        "test": test,
    }
    return sets, trainval_plain.classes


def get_loader(ds, shuffle):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                      generator=g, pin_memory=(DEVICE.type == "cuda"))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_model():
    """
    MobileNetV3-Small pre-trained on ImageNet-1K.
    Its classifier is: Linear(576,1024) - Hardswish - Dropout - Linear(1024,1000).
    The last layer (classifier[3]) is replaced with Linear(1024, 37).
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model, loader):
    """Returns true labels, predicted labels and mean cross-entropy loss."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        y_true += y.tolist()
        y_pred += out.argmax(1).tolist()
    return np.array(y_true), np.array(y_pred), total_loss / len(y_true)


# ------------------------------------------------------------
# Saving helpers
# ------------------------------------------------------------
def environment_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights": str(WEIGHTS),
    }
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_predictions(path, test_ds, y_true, y_pred):
    """One row per test image: file name, true label, predicted label."""
    names = get_image_names(test_ds)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred"])
        for n, t, p in zip(names, y_true, y_pred):
            w.writerow([n, int(t), int(p)])
