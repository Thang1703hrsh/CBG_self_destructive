"""SEAM loss, reversible perturbation, and FP32 AdamW with CPU optimizer state."""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def masked_causal_ce(logits, labels, mask):
    # Logits at t predict labels at t+1. Mask targets only, never logits.
    targets = labels[..., 1:].clone()
    targets.masked_fill_(~mask[..., 1:].bool(), -100)
    if not (targets != -100).any():
        raise ValueError("SEAM loss has no selected non-padding target tokens")
    return F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                           targets.reshape(-1))


def collate_seam_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


@contextmanager
def parameter_perturbation(model, grad_1, grad_2, norm_1, norm_2, cosine, epsilon,
                           storage_device='cpu'):
    if not all(math.isfinite(float(x)) for x in (norm_1, norm_2, cosine, epsilon)):
        raise FloatingPointError("Nonfinite SEAM perturbation input")
    if float(norm_1) <= 0 or float(norm_2) <= 0 or epsilon <= 0:
        raise ValueError("SEAM perturbation requires positive gradient norms and epsilon")
    original = {}
    changed, total, squared_norm = 0, 0, 0.0
    model.zero_grad(set_to_none=True)
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Snapshot on CPU in the corrected mode; do not duplicate the full
                # FP32 model on its GPU or replace Parameter storage on restoration.
                original[name] = param.detach().to(storage_device, copy=True)
                direction = grad_1[name].float() / norm_1 - cosine * grad_2[name].float() / norm_2
                param.add_(direction.to(param.device), alpha=epsilon)
                delta = param.detach().float() - original[name].to(param.device).float()
                if not torch.isfinite(delta).all():
                    raise FloatingPointError(f"Nonfinite perturbation: {name}")
                changed += torch.count_nonzero(delta).item()
                total += delta.numel()
                squared_norm += delta.square().sum(dtype=torch.float64).item()
                del direction, delta
        stats = {'changed_fraction': changed / total if total else 0.0,
                 'actual_norm': math.sqrt(squared_norm), 'epsilon': epsilon}
        if changed == 0:
            raise FloatingPointError("SEAM perturbation rounded entirely to zero; use FP32 training")
        yield stats
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.copy_(original[name].to(param.device))
        model.zero_grad(set_to_none=True)


class CPUAdamW(torch.optim.Optimizer):
    """Unquantized AdamW: FP32 model parameters, FP32 moments stored on CPU.

    Transfers one parameter at a time for the update. This trades CPU RAM and
    transfer time for GPU memory; no persistent rounded BF16 shadow weights.
    """

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for param in group['params']:
                if param.grad is None:
                    continue
                if param.dtype != torch.float32 or param.grad.is_sparse:
                    raise ValueError("CPUAdamW requires dense gradients and FP32 model parameters")
                gradient = param.grad.detach().to('cpu', dtype=torch.float32)
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite gradient in CPUAdamW")
                state = self.state[param]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(gradient),
                                 exp_avg_sq=torch.zeros_like(gradient))
                state['step'] += 1
                avg, sq = state['exp_avg'], state['exp_avg_sq']
                avg.lerp_(gradient, 1 - beta1)
                sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = sq.sqrt().div_(math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                value = param.detach().to('cpu', copy=True)
                value.mul_(1 - group['lr'] * group['weight_decay'])
                value.addcdiv_(avg, denominator, value=-group['lr'] / (1 - beta1 ** state['step']))
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite parameter update in CPUAdamW")
                param.copy_(value.to(param.device))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict normally places moments beside the parameter.
        for state in self.state.values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key] = state[key].to('cpu', dtype=torch.float32)
