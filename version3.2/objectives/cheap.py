# objectives/cheap.py
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("cheap_obj", logfile="logs/cheap.log")


# =============================================================================
# Fast graph-based param counter (no model build required)
# =============================================================================

def count_params_from_graph(graph) -> int:
    """
    O(n) parameter estimate directly from graph node metadata.
    No CompiledModel is built.  Used for fast KDE filtering before training.
    """
    total = 0
    for node in graph.nodes.values():
        p = node.params
        op = node.op_type
        if op == "conv":
            ic  = p.get("in_channels", 1)
            oc  = p.get("out_channels", 1)
            k   = p.get("kernel_size", 3)
            total += ic * oc * k * k + oc          # weight + bias
        elif op == "sep_conv":
            ic  = p.get("in_channels", 1)
            oc  = p.get("out_channels", 1)
            k   = p.get("kernel_size", 3)
            # depthwise: ic*k*k, pointwise: ic*oc, both biases
            total += ic * k * k + ic * oc + oc
        elif op == "bn":
            f = p.get("num_features", 1)
            total += f * 2                          # weight + bias
        elif op in ("fc", "linear"):
            in_f  = p.get("in_features", 1)
            out_f = p.get("out_features", 1)
            total += in_f * out_f + out_f
    return max(total, 1)


# =============================================================================
# Model-based param counter (exact, requires built model)
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    if model is None:
        return int(1e9)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# FLOPs estimator (requires built model + thop)
# =============================================================================

def estimate_flops(model: nn.Module, input_size=(1, 3, 32, 32)) -> float:
    if model is None:
        return float("inf")

    try:
        from thop import profile
    except ImportError:
        logger.warning("thop not installed → FLOPs reported as inf. pip install thop")
        return float("inf")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    dummy = torch.zeros(*input_size, device=device)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            if macs == 0:
                macs = 1.0
        except Exception as e:
            logger.error("FLOPs estimation failed: %s", e)
            return float("inf")

    # Clean up thop buffers so they don't pollute state_dicts
    _scrub_thop_buffers(model)
    return float(macs)


def _scrub_thop_buffers(model: nn.Module):
    """Remove thop profiling buffers injected into every module."""
    for m in model.modules():
        for key in ("total_ops", "total_params", "profile_ops", "profile_params"):
            m._buffers.pop(key, None)


def clean_state_dict(sd: dict) -> dict:
    """
    Remove thop artefacts + NaN tensors from a state dict.
    Used by workers when loading inherited weights.
    """
    cleaned = {}
    for k, v in sd.items():
        if any(tag in k for tag in ("total_ops", "total_params", "profile")):
            continue
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                continue          # drop corrupted tensors
            cleaned[k] = v.contiguous().clone()
        else:
            cleaned[k] = v
    return cleaned