"""Chunked per-example Linear-weight contractions for the specified CBG space."""
from collections import OrderedDict

import torch
from torch import nn


def spec_linear_modules(model, block_path="model.layers"):
    """Select only Linear weights inside transformer blocks, never the LM head."""
    try:
        blocks = model.get_submodule(block_path)
    except AttributeError as error:
        raise ValueError(f"transformer block container {block_path!r} not found") from error
    result = OrderedDict()
    seen = set()
    for name, module in blocks.named_modules():
        if isinstance(module, nn.Linear):
            if not module.weight.requires_grad:
                raise ValueError("all Linear weights in U must be differentiable")
            if id(module.weight) in seen:
                raise ValueError("shared Linear weights in U require cross-invocation contractions")
            seen.add(id(module.weight))
            result[f"{block_path}.{name}.weight"] = module
    if not result:
        raise ValueError("no Linear weights in the transformer blocks")
    return result


class SpecLinearHooks:
    """Two backward passes on one joint forward, without retaining full G_e.

    Each module must run once per forward. Reject weight reuse/checkpoint replay
    rather than silently omit cross terms between calls. Safety's two sequences
    are combined BEFORE any Gram or quadratic contraction.
    """
    def __init__(self, modules, seq_to_example, attention_mask, *, xi,
                 max_bytes=64 * 1024 * 1024, damping="relative"):
        self.modules = modules
        self.ids = seq_to_example
        self.mask = attention_mask.bool()
        self.E = int(seq_to_example.max()) + 1
        self.xi, self.max_bytes, self.damping = xi, max_bytes, damping
        self.handles, self.tensor_handles, self.seen = [], [], set()
        # The first executed admissible Linear input is an activation upstream
        # of every hooked Linear output in an ordinary transformer stack.
        # Differentiating to this activation traverses every output hook while
        # avoiding materialization of a discarded full-model weight gradient.
        self.backward_target = None
        self.mode = "gram"
        self.K = torch.zeros(self.E, self.E, device=self.ids.device, dtype=torch.float64)
        self.w, self.d_lambda = OrderedDict(), OrderedDict()

    def __enter__(self):
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._forward(name)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles + self.tensor_handles:
            handle.remove()
        self.handles.clear()
        self.tensor_handles.clear()

    def _forward(self, name):
        def capture(module, inputs, output):
            if name in self.seen:
                raise ValueError("CBG hook saw a reused Linear; checkpointing/reused weights unsupported")
            self.seen.add(name)
            source = inputs[0]
            if self.backward_target is None:
                self.backward_target = source
            x = source.detach()
            if x.ndim != 3 or x.shape[:2] != self.mask.shape:
                raise ValueError("CBG Linear hooks require unpacked [sequence, token, feature] inputs")
            self.tensor_handles.append(output.register_hook(
                lambda delta: self._backward(name, module, x, delta)))
        return capture

    def materialize(self, cw, A):
        if set(self.modules) != self.seen:
            raise ValueError("some admissible Linear modules were not evaluated")
        self.mode, self.cw, self.A = "directions", cw, A

    @torch.no_grad()
    def _backward(self, name, module, x, delta):
        dtype = torch.float64 if module.weight.dtype == torch.float64 else torch.float32
        # Mask inputs, not just response tokens: prompt activations also affect
        # response predictions through attention and MUST contribute to G_e.
        x = x.to(dtype) * self.mask.unsqueeze(-1)
        weight = module.weight.detach()
        batch, tokens, din = x.shape
        # Conservative bound for simultaneous sequence/example contraction and
        # FP64 Gram workspaces (activation storage is accounted for separately).
        bytes_per_row = 8 * (4 * self.E + 2 * batch + 4) * din + 8 * batch * tokens
        rows = self.max_bytes // bytes_per_row
        if rows < 1:
            raise ValueError("gram_max_bytes is too small for one output row")
        if self.mode == "directions":
            self.w[name] = torch.empty_like(weight)
            self.d_lambda[name] = torch.empty_like(weight)
        for start in range(0, weight.shape[0], rows):
            end = min(start + rows, weight.shape[0])
            d = delta[:, :, start:end].detach().to(dtype) * self.mask.unsqueeze(-1)
            sequence_grad = d.transpose(1, 2) @ x
            ge = x.new_zeros(self.E, end-start, din).index_add_(0, self.ids, sequence_grad)
            del sequence_grad, d
            lam_inv = (weight[start:end].to(dtype).square() + self.xi**2
                       if self.damping == "relative" else torch.ones_like(weight[start:end], dtype=dtype))
            if not bool((lam_inv > 0).all()):
                raise FloatingPointError("relative metric is singular: use positive xi for zero weights")
            flat = ge.flatten(1)
            if self.mode == "gram":
                weighted = flat.double() * lam_inv.flatten().double().sqrt()
                self.K.add_(weighted @ weighted.T)
            else:
                self.w[name][start:end] = (self.cw.to(flat) @ flat).reshape_as(lam_inv) * lam_inv
                if self.damping == "relative":
                    diag = (flat * (self.A.to(flat) @ flat)).sum(0).reshape_as(lam_inv)
                    self.d_lambda[name][start:end] = -2 * weight[start:end] * diag
                else:
                    self.d_lambda[name][start:end].zero_()
