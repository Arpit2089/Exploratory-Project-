# objectives/cheap.py
import torch
from utils.logger import get_logger

logger = get_logger("cheap_obj", logfile="logs/cheap_obj.log")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameter count: total=%d trainable=%d", total, trainable)
    return total

def estimate_flops(model, input_size=(1,3,32,32), device='cpu'):
    """
    Rough flop estimator using forward hooks for Conv2d and Linear.
    Counts multiply-adds as 1 op (matching paper's 'mult-adds' measure approximately).
    This is approximate and intended for relative comparisons.
    """
    model = model.to(device)
    hooks = []
    flops = {'total': 0}

    def conv_hook(self, inp, out):
        # inp: tuple (x,)
        input = inp[0]
        batch_size, in_c, in_h, in_w = input.shape
        out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        # multiply-adds per output element
        ops_per_element = kernel_ops
        this_flops = ops_per_element * out_c * out_h * out_w
        flops['total'] += this_flops
        logger.debug("Conv layer flops: out_c=%d out_h=%d out_w=%d kernel_ops=%s -> %d",
                     out_c, out_h, out_w, kernel_ops, this_flops)

    def linear_hook(self, inp, out):
        input = inp[0]
        batch_size = input.shape[0]
        weight_ops = self.weight.numel()
        flops['total'] += weight_ops
        logger.debug("Linear layer flops: weight_ops=%d", weight_ops)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        try:
            fake = torch.zeros(*input_size).to(device)
            _ = model(fake)
        except Exception as e:
            logger.exception("Failed to run flop estimation forward pass")
            for h in hooks: h.remove()
            raise
    for h in hooks: h.remove()
    logger.info("Estimated FLOPs (approx, mult-adds): %d", flops['total'])
    return flops['total']
