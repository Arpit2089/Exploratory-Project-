# data/loader_factory.py
# =============================================================================
# Single entry-point for data loading.
#
# get_loaders(cfg)           — main process: uses cfg.NUM_DATALOADER_WORKERS
#                              and cfg.PIN_MEMORY for fast GPU data feeds.
#
# get_loaders_for_worker(cfg) — CPU subprocess workers: always num_workers=0
#                               and pin_memory=False (nested MP is unsafe).
# =============================================================================
import torch


def get_loaders(cfg, split_test: bool = True):
    """
    Return (train_loader, val_loader[, test_loader]) for the main process.

    Uses cfg.NUM_DATALOADER_WORKERS prefetch workers and cfg.PIN_MEMORY for
    fast CPU→GPU data transfers when CUDA is available.
    """
    ds  = cfg.TARGET_DATASET
    bs  = cfg.BATCH_SIZE
    fdm = cfg.FAST_DEV_MODE

    # pin_memory is only useful when CUDA is available
    pin = cfg.PIN_MEMORY and torch.cuda.is_available()
    nw  = cfg.NUM_DATALOADER_WORKERS if torch.cuda.is_available() else 0

    if ds == "CIFAR-100":
        from data.cifar100 import get_cifar100_loaders
        return get_cifar100_loaders(
            batch_size=bs, num_workers=nw, pin_memory=pin,
            split_test=split_test, fast_dev_mode=fdm,
        )
    elif ds == "IMAGENET":
        from data.imagenet import get_imagenet_loaders
        return get_imagenet_loaders(
            batch_size=bs, num_workers=nw, pin_memory=pin,
            split_test=split_test, fast_dev_mode=fdm,
        )
    else:  # default: CIFAR-10
        from data.cifar10 import get_cifar_loaders
        return get_cifar_loaders(
            batch_size=bs, num_workers=nw, pin_memory=pin,
            split_test=split_test, fast_dev_mode=fdm,
        )


def get_loaders_for_worker(cfg):
    """
    Returns (train_loader, val_loader) for use inside a CPU subprocess worker.
    Always uses num_workers=0 and pin_memory=False to avoid nested-MP crashes.
    """
    ds  = cfg.TARGET_DATASET
    bs  = cfg.BATCH_SIZE
    fdm = cfg.FAST_DEV_MODE

    if ds == "CIFAR-100":
        from data.cifar100 import get_cifar100_loaders
        result = get_cifar100_loaders(
            batch_size=bs, num_workers=0, pin_memory=False,
            split_test=True, fast_dev_mode=fdm,
        )
    elif ds == "IMAGENET":
        from data.imagenet import get_imagenet_loaders
        result = get_imagenet_loaders(
            batch_size=bs, num_workers=0, pin_memory=False,
            split_test=True, fast_dev_mode=fdm,
        )
    else:
        from data.cifar10 import get_cifar_loaders
        result = get_cifar_loaders(
            batch_size=bs, num_workers=0, pin_memory=False,
            split_test=True, fast_dev_mode=fdm,
        )
    # result is (train, val, test) — we only need train and val
    return result[0], result[1]