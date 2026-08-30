"""Focused 2D SOAP optimizer with exact eigendecomposition every step.

The update follows the official SOAP implementation for matrix parameters:
Shampoo statistics define left and right eigenbases, Adam moments are updated
in that rotated basis, and the normalized update is projected back.  Unlike
the preliminary reference implementation's periodic QR refresh, this variant
uses ``torch.linalg.eigh`` for every basis refresh.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _descending_eigenvectors(matrix: Tensor) -> Tensor:
    matrix = (matrix.float() + matrix.float().mT) * 0.5
    try:
        _, eigenvectors = torch.linalg.eigh(matrix)
    except RuntimeError:
        _, eigenvectors = torch.linalg.eigh(matrix.double())
        eigenvectors = eigenvectors.float()
    return eigenvectors.flip(1)


def _project_to_basis(matrix: Tensor, left: Tensor, right: Tensor) -> Tensor:
    return left.mT @ matrix.float() @ right


def _project_from_basis(matrix: Tensor, left: Tensor, right: Tensor) -> Tensor:
    return left @ matrix.float() @ right.mT


class SOAPWithAuxAdam(torch.optim.Optimizer):
    """SOAP for selected matrices and AdamW for auxiliary parameters."""

    def __init__(
        self,
        soap_params,
        adam_params,
        *,
        lr: float,
        adam_lr: float,
        betas=(0.95, 0.95),
        adam_betas=(0.9, 0.999),
        shampoo_beta: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        soap_params, adam_params = list(soap_params), list(adam_params)
        groups = []
        if soap_params:
            groups.append({"params": soap_params, "lr": lr, "use_soap": True})
        if adam_params:
            groups.append({"params": adam_params, "lr": adam_lr,
                           "use_soap": False})
        defaults = dict(
            lr=lr,
            betas=betas,
            adam_betas=adam_betas,
            shampoo_beta=betas[1] if shampoo_beta is None else shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_soap"]:
                self._step_soap_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_adam_group(self, group):
        beta1, beta2 = group["adam_betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            average, square_average = state["exp_avg"], state["exp_avg_sq"]
            average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            square_average.mul_(beta2).addcmul_(gradient, gradient,
                                                value=1.0 - beta2)
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            denominator = square_average.sqrt().div_(correction2 ** 0.5)
            denominator.add_(group["eps"])
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.addcdiv_(average, denominator,
                               value=-group["lr"] / correction1)

    def _step_soap_group(self, group):
        beta1, beta2 = group["betas"]
        shampoo_beta = group["shampoo_beta"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.ndim != 2:
                raise ValueError("SOAP routing must contain only matrices")
            gradient = parameter.grad.float()
            state = self.state[parameter]

            if "left_factor" not in state:
                rows, columns = gradient.shape
                state["step"] = 0
                state["left_factor"] = torch.zeros(
                    rows, rows, device=gradient.device, dtype=torch.float32)
                state["right_factor"] = torch.zeros(
                    columns, columns, device=gradient.device, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
                self._update_factors_and_basis(
                    gradient, state, shampoo_beta, rotate_average=False)
                continue

            left, right = state["left_basis"], state["right_basis"]
            projected = _project_to_basis(gradient, left, right)
            average, square_average = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            average.mul_(beta1).add_(projected, alpha=1.0 - beta1)
            square_average.mul_(beta2).addcmul_(projected, projected,
                                                value=1.0 - beta2)
            correction1 = 1.0 - beta1 ** state["step"]
            correction2 = 1.0 - beta2 ** state["step"]
            normalized = average / correction1
            denominator = (square_average / correction2).sqrt().add_(group["eps"])
            update = _project_from_basis(normalized / denominator, left, right)

            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
            self._update_factors_and_basis(
                gradient, state, shampoo_beta, rotate_average=True)

    @staticmethod
    def _update_factors_and_basis(
        gradient: Tensor,
        state: dict,
        shampoo_beta: float,
        *,
        rotate_average: bool,
    ) -> None:
        state["left_factor"].lerp_(gradient @ gradient.mT,
                                   1.0 - shampoo_beta)
        state["right_factor"].lerp_(gradient.mT @ gradient,
                                    1.0 - shampoo_beta)
        if rotate_average:
            average_original = _project_from_basis(
                state["exp_avg"], state["left_basis"], state["right_basis"])
        state["left_basis"] = _descending_eigenvectors(state["left_factor"])
        state["right_basis"] = _descending_eigenvectors(state["right_factor"])
        if rotate_average:
            state["exp_avg"].copy_(_project_to_basis(
                average_original, state["left_basis"], state["right_basis"]))

