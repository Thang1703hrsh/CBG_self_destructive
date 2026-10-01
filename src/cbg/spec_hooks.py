"""Chunked per-example Linear-weight contractions for the specified CBG space."""
from collections import OrderedDict
from contextlib import contextmanager

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
        if hasattr(module, "lora_A"):
            raise NotImplementedError("v2 LoRA requires effective-weight geometry and adapter chain rules; plain Linear hooks cannot implement it")
        if isinstance(module, nn.Linear):
            if id(module.weight) in seen:
                raise ValueError("shared Linear weights in U require cross-invocation contractions")
            seen.add(id(module.weight))
            result[f"{block_path}.{name}.weight"] = module
    if not result:
        raise ValueError("no Linear weights in the transformer blocks")
    return result


@contextmanager
def _tf32(enabled):
    """TF32 is exact for products of BF16-stored captures (8 <= 10 mantissa bits)."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class SpecLinearHooks:
    """Capture (X, Delta) for every admissible Linear, then contract without a graph.

    A *call* is one forward/backward of a set of sequences (the whole joint
    batch, or one micro-batch of it). Each module must run exactly once per
    call; reuse inside a call (checkpoint replay, tied weights) is rejected
    rather than silently dropping cross terms. Because examples never interact
    in the forward pass, per-example gradients of different calls are simply
    summed into the same example rows, and the Gram is contracted only after
    every call has been captured. Safety's two sequences are combined BEFORE any
    Gram or quadratic contraction.
    """
    def __init__(self, modules, seq_to_example=None, attention_mask=None, *, xi,
                 max_bytes=64 * 1024 * 1024, damping="relative",
                 capture_dtype="model", offload=False, num_examples=None,
                 contraction_tf32=False):
        self.modules = modules
        self.xi, self.max_bytes, self.damping = xi, max_bytes, damping
        self.capture_dtype, self.offload = capture_dtype, offload
        self.contraction_tf32 = contraction_tf32
        self.handles, self.tensor_handles = [], []
        self.calls = []            # (seq_to_example, attention_mask) per call
        self.call = None
        self.seen = set()          # (name, call) pairs that ran forward
        # The first executed admissible Linear input of the current call is an
        # activation upstream of every hooked Linear output in an ordinary
        # transformer stack. Differentiating to it traverses every output hook
        # without materializing a discarded full-model weight gradient.
        self.backward_target = None
        self.captures = OrderedDict()   # (name, call) -> (x, delta)
        self.w, self.d_lambda = OrderedDict(), OrderedDict()
        self.d_lambda_sq = 0.
        self._K = None
        if seq_to_example is not None:
            self.E = int(seq_to_example.max()) + 1
            self.begin_call(seq_to_example, attention_mask)
        elif num_examples is None:
            raise ValueError("pass seq_to_example or num_examples")
        else:
            self.E = int(num_examples)
        self._device = (seq_to_example.device if seq_to_example is not None
                        else next(iter(modules.values())).weight.device)

    def begin_call(self, seq_to_example, attention_mask):
        """Declare the sequences of the next forward (global example indices)."""
        if seq_to_example.ndim != 1 or attention_mask.shape[0] != len(seq_to_example):
            raise ValueError("one example index per sequence is required")
        if len(seq_to_example) and int(seq_to_example.max()) >= self.E:
            raise ValueError("example index outside the declared example count")
        self.calls.append((seq_to_example, attention_mask.bool()))
        self.call = len(self.calls) - 1
        self.backward_target = None
        self._K = None

    def __enter__(self):
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_pre_hook(self._ensure_input_grad))
            self.handles.append(module.register_forward_hook(self._forward(name)))
        return self

    @staticmethod
    def _ensure_input_grad(module, inputs):
        # Geometry must span full U even when T is a frozen-weight subset.
        # This creates a temporary activation leaf, never unfreezes weights.
        if not inputs[0].requires_grad:
            # Preserve the shared activation identity across q/k/v branches.
            # Returning separate detached leaves would prune sibling branches
            # when differentiating to the first captured activation.
            inputs[0].requires_grad_(True)

    def __exit__(self, *exc):
        for handle in self.handles + self.tensor_handles:
            handle.remove()
        self.handles.clear()
        self.tensor_handles.clear()

    def _forward(self, name):
        def capture(module, inputs, output):
            call = self.call
            if call is None:
                raise ValueError("begin_call must precede a captured forward")
            if (name, call) in self.seen:
                raise ValueError("CBG hook saw a reused Linear; checkpointing/reused weights unsupported")
            self.seen.add((name, call))
            source = inputs[0]
            if self.backward_target is None:
                self.backward_target = source
            x = source.detach()
            if x.ndim != 3 or x.shape[:2] != self.calls[call][1].shape:
                raise ValueError("CBG Linear hooks require unpacked [sequence, token, feature] inputs")
            dtype = torch.bfloat16 if self.capture_dtype == "bf16" and x.dtype != torch.float64 else x.dtype
            x = x.to(device="cpu" if self.offload else x.device, dtype=dtype, copy=True)
            def backward(delta):
                # A repeated backward of the same graph overwrites identical values.
                self.captures[(name, call)] = (x, delta.detach().to(device=x.device, dtype=dtype, copy=True))
                self._K = None
            self.tensor_handles.append(output.register_hook(backward))
        return capture

    def captured_names(self):
        return {name for name, _ in self.captures}

    def _check_complete(self):
        expected = {(name, call) for name in self.modules for call in range(len(self.calls))}
        if self.seen != expected:
            raise ValueError("some admissible Linear modules were not evaluated")
        if set(self.captures) != expected:
            raise ValueError("center backward did not capture all admissible modules")

    @property
    def K(self):
        if self._K is None:
            self._check_complete()
            self._K = torch.zeros(self.E, self.E, device=self._device, dtype=torch.float64)
            for name, module in self.modules.items():
                self._contract(name, module, "gram")
        return self._K

    def materialize(self, cw, A, *, d_lambda_scale=None, compute_d_lambda=True):
        """w = Lambda^{-1} C'b and d_Lambda = -2 theta . diag(C'AC) (A = -G).

        With d_lambda_scale, scale*d_Lambda is added to each weight's .grad
        instead of being stored (one parameter-sized tree less).
        """
        self._check_complete()
        self.cw, self.A = cw, A
        self.d_lambda_scale = d_lambda_scale
        self.compute_d_lambda = compute_d_lambda
        self.d_lambda_sq = 0.
        for name, module in self.modules.items():
            self._contract(name, module, "directions")

    def _call_tensors(self, call, device, dtype):
        ids, mask = self.calls[call]
        return ids.to(device), mask.to(device=device).unsqueeze(-1).to(dtype)

    @torch.no_grad()
    def _contract(self, name, module, mode):
        weight = module.weight.detach()
        device = weight.device
        dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        din = weight.shape[1]
        parts = [(call,) + self.captures[(name, call)] for call in range(len(self.calls))]
        widest = max(x.shape[0] for _, x, _ in parts)
        tokens = max(x.shape[1] for _, x, _ in parts)
        # Conservative bound for simultaneous sequence/example contraction and
        # FP64 Gram workspaces (activation storage is accounted for separately).
        bytes_per_row = 8 * (4 * self.E + 2 * widest + 4) * din + 8 * widest * tokens
        rows = self.max_bytes // bytes_per_row
        if rows < 1:
            raise ValueError("gram_max_bytes is too small for one output row")
        if mode == "directions":
            self.w[name] = torch.empty_like(weight)
            if self.compute_d_lambda and self.d_lambda_scale is None:
                self.d_lambda[name] = torch.empty_like(weight)
        # Mask inputs, not just response tokens: prompt activations also affect
        # response predictions through attention and MUST contribute to G_e.
        inputs = []
        for call, x, delta in parts:
            ids, mask = self._call_tensors(call, device, dtype)
            inputs.append((ids, mask, x.to(device=device, dtype=dtype) * mask, delta))
        exact_tf32 = self.contraction_tf32 and all(d.dtype == torch.bfloat16 for _, _, d in parts)
        for start in range(0, weight.shape[0], rows):
            end = min(start + rows, weight.shape[0])
            ge = torch.zeros(self.E, end - start, din, device=device, dtype=dtype)
            with _tf32(exact_tf32):
                for ids, mask, x, delta in inputs:
                    d = delta[:, :, start:end].to(device=device, dtype=dtype) * mask
                    ge.index_add_(0, ids, d.transpose(1, 2) @ x)
                    del d
            lam_inv = (weight[start:end].to(dtype).square() + self.xi**2
                       if self.damping == "relative" else torch.ones_like(weight[start:end], dtype=dtype))
            if not bool((lam_inv > 0).all()):
                raise FloatingPointError("relative metric is singular: use positive xi for zero weights")
            flat = ge.flatten(1)
            if mode == "gram":
                weighted = flat.double() * lam_inv.flatten().double().sqrt()
                self._K.add_((weighted @ weighted.T).to(self._K.device))
                del weighted
            else:
                self.w[name][start:end] = (self.cw.to(flat) @ flat).reshape_as(lam_inv) * lam_inv
                if self.compute_d_lambda:
                    if self.damping == "relative":
                        diag = (flat * (self.A.to(flat) @ flat)).sum(0).reshape_as(lam_inv)
                        value = -2 * weight[start:end].to(dtype) * diag
                    else:
                        value = torch.zeros_like(lam_inv)
                    self.d_lambda_sq += float(value.double().square().sum())
                    if self.d_lambda_scale is None:
                        self.d_lambda[name][start:end] = value
                    else:
                        if module.weight.grad is None:
                            module.weight.grad = torch.zeros_like(weight)
                        module.weight.grad[start:end].add_(value.to(weight.dtype), alpha=self.d_lambda_scale)
            del ge, flat
