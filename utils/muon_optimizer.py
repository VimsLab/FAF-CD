import torch


def _quantize_int8_dynamic(tensor: torch.Tensor):
    tensor_fp32 = tensor.to(torch.float32)
    max_abs = tensor_fp32.abs().amax()
    if not torch.isfinite(max_abs) or max_abs.item() == 0.0:
        scale = torch.ones(1, device=tensor.device, dtype=torch.float32)
        q = torch.zeros_like(tensor, dtype=torch.int8, memory_format=torch.preserve_format)
        return q, scale

    scale = (max_abs / 127.0).to(torch.float32).reshape(1)
    q = torch.clamp(torch.round(tensor_fp32 / scale), -127, 127).to(torch.int8)
    return q, scale


def _dequantize_int8_dynamic(q_tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return q_tensor.to(torch.float32).mul(scale).to(dtype=dtype)


def zeropower_via_newtonschulz5(grad_matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration used by Muon to orthogonalize matrix updates.
    Adapted from KellerJordan/Muon.
    """
    if grad_matrix.ndim < 2:
        raise ValueError(f"Expected tensor with ndim>=2, got {grad_matrix.ndim}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad_matrix.to(torch.bfloat16)

    transposed = False
    if x.size(-2) > x.size(-1):
        x = x.mT
        transposed = True

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(int(steps)):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x

    if transposed:
        x = x.mT
    return x


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * (max(1.0, float(update.size(-2)) / float(update.size(-1))) ** 0.5)
    return update


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas,
    eps: float,
) -> torch.Tensor:
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1.0 - beta2)
    exp_avg_corr = exp_avg / (1.0 - beta1**step)
    exp_avg_sq_corr = exp_avg_sq / (1.0 - beta2**step)
    return exp_avg_corr / (exp_avg_sq_corr.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Single-optimizer Muon + AdamW implementation.

    Param group contract:
    - Muon groups: {"params", "use_muon", "lr", "momentum", "weight_decay", "lr_mult"}
    - Adam groups: {"params", "use_muon", "lr", "betas", "eps", "weight_decay", "lr_mult"}
    """

    def __init__(
        self,
        param_groups,
        ns_steps: int = 5,
        nesterov: bool = True,
        muon_quantization: str = "none",
        adam_state_precision: str = "fp32",
    ):
        if not isinstance(param_groups, (list, tuple)) or not param_groups:
            raise ValueError("param_groups must be a non-empty list")

        norm_quantization = str(muon_quantization or "none").strip().lower()
        if norm_quantization not in {"none", "dynamic"}:
            raise ValueError(f"Unsupported muon_quantization: {muon_quantization}")

        norm_state_precision = str(adam_state_precision or "fp32").strip().lower()
        if norm_state_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported muon_adamw_state_precision: {adam_state_precision}")

        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param group must define use_muon")
            group["lr_mult"] = float(group.get("lr_mult", 1.0))
            if group["use_muon"]:
                group["lr"] = float(group.get("lr", 0.02))
                group["momentum"] = float(group.get("momentum", 0.95))
                group["weight_decay"] = float(group.get("weight_decay", 0.0))
            else:
                group["lr"] = float(group.get("lr", 3e-4))
                group["betas"] = tuple(group.get("betas", (0.9, 0.999)))
                group["eps"] = float(group.get("eps", 1e-8))
                group["weight_decay"] = float(group.get("weight_decay", 0.0))

        defaults = dict(
            ns_steps=int(ns_steps),
            nesterov=bool(nesterov),
            muon_quantization=norm_quantization,
            adam_state_precision=norm_state_precision,
        )
        super().__init__(param_groups, defaults)

    def _adam_state_dtype(self, param: torch.Tensor) -> torch.dtype:
        precision = self.defaults["adam_state_precision"]
        if precision == "fp32":
            return torch.float32
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        return param.dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ns_steps = int(self.defaults["ns_steps"])
        nesterov = bool(self.defaults["nesterov"])
        muon_quantization = str(self.defaults.get("muon_quantization", "none")).lower()

        for group in self.param_groups:
            use_muon = bool(group["use_muon"])
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))

            if use_muon:
                beta = float(group["momentum"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients")
                    state = self.state[p]
                    if muon_quantization == "dynamic":
                        if "momentum_buffer_q" not in state or "momentum_buffer_scale" not in state:
                            q_init = torch.zeros_like(p, dtype=torch.int8, memory_format=torch.preserve_format)
                            scale_init = torch.ones(1, device=p.device, dtype=torch.float32)
                            state["momentum_buffer_q"] = q_init
                            state["momentum_buffer_scale"] = scale_init
                        momentum_buffer = _dequantize_int8_dynamic(
                            state["momentum_buffer_q"],
                            state["momentum_buffer_scale"],
                            dtype=grad.dtype,
                        )
                    else:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        momentum_buffer = state["momentum_buffer"]

                    update = muon_update(
                        grad=grad,
                        momentum_buffer=momentum_buffer,
                        beta=beta,
                        ns_steps=ns_steps,
                        nesterov=nesterov,
                    )
                    if muon_quantization == "dynamic":
                        momentum_q, momentum_scale = _quantize_int8_dynamic(momentum_buffer)
                        state["momentum_buffer_q"] = momentum_q
                        state["momentum_buffer_scale"] = momentum_scale

                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p).to(dtype=p.dtype), alpha=-lr)
            else:
                betas = group["betas"]
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam path does not support sparse gradients")
                    state = self.state[p]
                    if "step" not in state:
                        state_dtype = self._adam_state_dtype(p)
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype, memory_format=torch.preserve_format)
                    state["step"] += 1
                    update = adam_update(
                        grad=grad.to(state["exp_avg"].dtype),
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        step=state["step"],
                        betas=betas,
                        eps=eps,
                    )
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr)
        return loss
