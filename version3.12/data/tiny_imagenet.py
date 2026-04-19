# data/tiny_imagenet.py
# =============================================================================
# Tiny ImageNet data loader.
#
#   - 200 classes, 64×64 RGB images
#   - 100,000 train images  (500 per class)
#   - 10,000  val   images  (50  per class)
#
# The dataset is automatically downloaded and prepared the first time it
# is requested.  The val folder is reorganised from the flat layout that
# ships in the zip into the ImageFolder-compatible class-subfolder layout
# so that torchvision.datasets.ImageFolder works out of the box.
#
# Download source: http://cs231n.stanford.edu/tiny-imagenet-200.zip (~237 MB)
#
# Windows fixes applied:
#   - prefetch_factor=2  : overlaps disk I/O with GPU compute (big speedup)
#   - drop_last=True     : avoids variable-size last batches hurting AMP
#   - persistent_workers : kept alive between epochs (avoids respawn cost)
# =============================================================================

import os
import shutil
import zipfile
import urllib.request
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

_TINY_URL   = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_ZIP_NAME   = "tiny-imagenet-200.zip"
_DATA_DIR   = "./data/tiny-imagenet-200"


# ---------------------------------------------------------------------------
# Download + preparation
# ---------------------------------------------------------------------------

def _show_progress(block_count, block_size, total_size):
    """Simple progress callback for urllib.request.urlretrieve."""
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1e6
        tot = total_size  / 1e6
        print(f"\r  Downloading Tiny ImageNet: {mb:.1f}/{tot:.1f} MB "
              f"({pct:.1f}%)    ", end="", flush=True)


def _reorganise_val(val_dir: str):
    """
    The shipped val folder is flat:
        val/images/val_0.JPEG  ...  val_9999.JPEG
        val/val_annotations.txt

    We move images into class subfolders so ImageFolder works:
        val/<wnid>/val_N.JPEG
    """
    images_dir = os.path.join(val_dir, "images")
    ann_file   = os.path.join(val_dir, "val_annotations.txt")
    sentinel   = os.path.join(val_dir, "_reorganised")

    if os.path.exists(sentinel):
        return  # already done

    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Expected val annotation file at {ann_file}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Expected val images dir at {images_dir}")

    print("  Reorganising Tiny ImageNet val folder (one-time) …", flush=True)

    # Parse annotation file: filename  wnid  x1 y1 x2 y2
    img_to_class = {}
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_to_class[parts[0]] = parts[1]

    # Create class subdirectories and move images
    for img_name, wnid in img_to_class.items():
        class_dir = os.path.join(val_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(class_dir,  img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Remove now-empty images directory
    try:
        os.rmdir(images_dir)
    except OSError:
        pass

    # Write sentinel so we don't redo this on every run
    open(sentinel, "w").close()
    print("  Val folder reorganised.", flush=True)


def _download_and_prepare(root: str = "./data"):
    """Download Tiny ImageNet zip if needed, extract, and fix val layout."""
    zip_path  = os.path.join(root, _ZIP_NAME)
    data_path = os.path.join(root, "tiny-imagenet-200")
    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    # ── Step 1: Download ──────────────────────────────────────────────
    if not os.path.isdir(train_dir):
        os.makedirs(root, exist_ok=True)

        if not os.path.isfile(zip_path):
            print(f"  Tiny ImageNet not found. Downloading from:\n  {_TINY_URL}")
            print("  (~237 MB — this may take a few minutes)")
            try:
                urllib.request.urlretrieve(_TINY_URL, zip_path, _show_progress)
                print()  # newline after progress bar
            except Exception as e:
                # Clean up partial download
                try:
                    os.remove(zip_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Failed to download Tiny ImageNet: {e}\n"
                    "Please download manually from:\n"
                    f"  {_TINY_URL}\n"
                    f"and place the zip at: {zip_path}"
                ) from e
        else:
            print(f"  Found existing zip at {zip_path}, extracting …")

        # ── Step 2: Extract ───────────────────────────────────────────
        print(f"  Extracting to {root} …", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        print("  Extraction complete.", flush=True)

        # Remove zip to save disk space
        try:
            os.remove(zip_path)
        except OSError:
            pass

    # ── Step 3: Reorganise val ────────────────────────────────────────
    _reorganise_val(val_dir)


# ---------------------------------------------------------------------------
# Public loader function
# ---------------------------------------------------------------------------

def get_tiny_imagenet_loaders(batch_size=128, num_workers=4, pin_memory=False,
                               split_test=False, fast_dev_mode=False):
    """
    Returns Tiny ImageNet data loaders, optimised for Windows laptop GPU.

    Parameters
    ----------
    batch_size    : mini-batch size (128 for ≤6 GB VRAM, 256 for 8 GB+)
    num_workers   : prefetch workers (4 is ideal for most Windows laptops,
                    set 0 only if you get DataLoader RuntimeErrors)
    pin_memory    : pin CPU tensors for fast GPU transfers (True with CUDA)
    split_test    : if True, returns (train, val, test); else (train, val)
    fast_dev_mode : if True, uses ~2% of data for quick iteration / debugging
    """
    _download_and_prepare(root="./data")

    train_dir = os.path.join(_DATA_DIR, "train")
    val_dir   = os.path.join(_DATA_DIR, "val")

    # ── Transforms ────────────────────────────────────────────────────
    # Tiny ImageNet-specific normalisation statistics
    mean = (0.4802, 0.4481, 0.3975)
    std  = (0.2770, 0.2691, 0.2821)

    # FIX: Use RandomResizedCrop instead of RandomCrop+padding.
    # RandomResizedCrop simulates scale variation better for 64x64 images
    # and produces significantly more diverse augmentations, helping NAS
    # discover more robust architectures.
    # ColorJitter + RandAugment give the model harder examples = faster learning.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Random erase: zeroes out a random patch — acts as strong regularisation
        # and prevents over-fitting on the small 64x64 images.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    # Validation / test: no augmentation, just normalise
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Dataset objects ───────────────────────────────────────────────
    train_data_full = torchvision.datasets.ImageFolder(
        root=train_dir, transform=transform_train
    )
    val_data_full = torchvision.datasets.ImageFolder(
        root=val_dir, transform=transform_test
    )

    # ── DataLoader factory ────────────────────────────────────────────
    # prefetch_factor=2: each worker pre-fetches 2 batches ahead so the
    # GPU never stalls waiting for data. Big speedup on num_workers > 0.
    persistent = (num_workers > 0)
    prefetch   = 2 if num_workers > 0 else None

    def make_loader(dataset, shuffle, drop_last=False):
        loader_kwargs = dict(
            batch_size        = batch_size,
            shuffle           = shuffle,
            num_workers       = num_workers,
            pin_memory        = pin_memory,
            persistent_workers= persistent,
            drop_last         = drop_last,
        )
        # prefetch_factor is only valid when num_workers > 0
        if prefetch is not None:
            loader_kwargs["prefetch_factor"] = prefetch
        return DataLoader(dataset, **loader_kwargs)

    # ── Fast-dev mode: tiny subsets ───────────────────────────────────
    if fast_dev_mode:
        n_train = max(1, int(0.02 * len(train_data_full)))  # ~2000 samples
        n_val   = max(1, int(0.05 * len(val_data_full)))    # ~500  samples
        np.random.seed(42)
        train_idx = np.random.choice(len(train_data_full), n_train, replace=False).tolist()
        val_idx   = np.random.choice(len(val_data_full),   n_val,   replace=False).tolist()
        train_data_full = Subset(train_data_full, train_idx)
        val_data_full   = Subset(val_data_full,   val_idx)

    if split_test:
        # Use the official val split as test; carve out 10% of train as val
        full_train = train_data_full
        n          = len(full_train)
        np.random.seed(42)
        indices    = np.random.permutation(n).tolist()

        if fast_dev_mode:
            split = max(1, int(0.8 * n))
        else:
            split = int(0.9 * n)

        train_idx = indices[:split]
        val_idx   = indices[split:]

        # We need the base dataset (not a Subset) for the val transform
        if isinstance(full_train, Subset):
            base_idxs = full_train.indices
            val_base  = torchvision.datasets.ImageFolder(
                root=train_dir, transform=transform_test
            )
            actual_train_idx = [base_idxs[i] for i in train_idx]
            actual_val_idx   = [base_idxs[i] for i in val_idx]
            train_set = Subset(full_train.dataset, actual_train_idx)
            val_set   = Subset(val_base,            actual_val_idx)
        else:
            val_base  = torchvision.datasets.ImageFolder(
                root=train_dir, transform=transform_test
            )
            train_set = Subset(full_train, train_idx)
            val_set   = Subset(val_base,   val_idx)

        # drop_last=True on train: avoids tiny final batches that can
        # cause instability with AMP (BatchNorm needs ≥2 samples).
        train_loader = make_loader(train_set,     shuffle=True,  drop_last=True)
        val_loader   = make_loader(val_set,       shuffle=False, drop_last=False)
        test_loader  = make_loader(val_data_full, shuffle=False, drop_last=False)
        return train_loader, val_loader, test_loader

    else:
        train_loader = make_loader(train_data_full, shuffle=True,  drop_last=True)
        val_loader   = make_loader(val_data_full,   shuffle=False, drop_last=False)
        return train_loader, val_loader