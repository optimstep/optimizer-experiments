"""Small Muon variants used by the GradCache training example.

Matrix parameters use either the polar update ``U V.T`` (standard Muon) or
the inverse-half spectral update ``U S**(-1/2) V.T``.  Parameters that are not
hidden matrices are handled by an auxiliary AdamW optimizer.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


INVERSE_SQRT_EXPRESS_COEFFS = (
    (7.24703301, -19.97085561, 14.29954465),
    (2.88515801, -2.15694901, 0.47267493),
    (2.03238790, -1.41557277, 0.39190485),
    (1.87608875, -1.25120918, 0.37512093),
    (1.87500000, -1.25000000, 0.37500000),
)

# Remez/Polar-Express stages for w -> 1 with
# w_next = a*w + b*w**5 + c*w**9, w in [1e-3**(1/4), 1.05**(1/4)].
# The last two stages use the locally third-order inverse-fourth-root step.
INVERSE_FOURTH_EXPRESS_COEFFS = (
    (3.243379374638676, -7.667175143756514, 5.636718755462006),
    (1.5960803445047933, -0.6023681833501827, 0.09846360112792833),
    (1.4200965549445874, -0.5694827267016853, 0.15020148745354303),
    (1.40625, -0.5625, 0.15625),
    (1.40625, -0.5625, 0.15625),
)

STABLE_EXPRESS_COEFFS = (
    (3.7735, -8.3649, 4.7397),
    (4.1015, -10.5015, 6.8262),
    (3.4766, -8.7202, 5.8348),
    (2.7722, -7.0218, 5.7043),
    (3.2507, -6.2916, 5.2491),
)


def subsampled_polar_polish(
    X: Tensor,
    fraction: float = 0.125,
    steps: int = 3,
    eta: float = 0.5,
) -> Tensor:
    """Polish the rows contributing most to ``||X X.T - I||_F``."""
    if steps <= 0 or fraction <= 0.0:
        return X
    if X.ndim != 2 or X.shape[0] > X.shape[1]:
        raise ValueError("polar polishing expects a wide two-dimensional matrix")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("polish fraction must be in (0, 1]")
    if not 0.0 < eta <= 0.5:
        raise ValueError("polish eta must be in (0, 0.5]")

    rows = X.shape[0]
    selected_rows = max(1, min(rows, round(rows * fraction)))
    gram = X @ X.mT
    score_gram = gram.float()
    score = (score_gram.square().sum(dim=-1)
             - 2.0 * score_gram.diagonal() + 1.0)
    indices = score.topk(selected_rows, largest=True, sorted=False).indices

    result = X.clone()
    for _ in range(steps):
        selected = result.index_select(0, indices)
        selected_gram = selected @ result.mT
        corrected = selected - eta * (selected_gram @ result - selected)
        result.index_copy_(0, indices, corrected)
    return result


def muon_polished(
    G: Tensor,
    eps: float = 1e-7,
    fraction: float = 0.125,
    steps: int = 3,
    eta: float = 0.5,
) -> Tensor:
    """Ordinary Muon followed by targeted restricted-row polishing."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = muon_polar(G, eps).float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    X = subsampled_polar_polish(X, fraction=fraction, steps=steps, eta=eta)
    if transposed:
        X = X.mT
    return X.to(original_dtype)


def muon_stable(G: Tensor) -> Tensor:
    """Stable Polar Express Muon without targeted polishing."""
    original_dtype = G.dtype
    X = G.float()
    X = X / X.norm().clamp_min(torch.finfo(X.dtype).tiny)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    for a, b, c in STABLE_EXPRESS_COEFFS:
        gram = X @ X.mT
        X = a * X + (b * gram + c * (gram @ gram)) @ X
    if transposed:
        X = X.mT
    return X.to(original_dtype)


def _gram_inverse_root(
    matrix: Tensor,
    power: int,
    coeffs,
    eps: float,
    restart_after: int = 2,
) -> Tensor:
    """Approximate a damped SPD inverse root with Gram-NS correction."""
    scale = matrix.norm().clamp_min(torch.finfo(matrix.dtype).tiny)
    A = matrix / scale
    identity = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A = A + eps * identity
    result = identity
    Q = identity
    R = A
    for stage, (a, b, c) in enumerate(coeffs, 1):
        R2 = R @ R
        F = a * identity + b * R + c * R2
        Q = Q @ F
        R = R @ torch.linalg.matrix_power(F, power)
        R = (R + R.mT) * 0.5
        if stage == restart_after:
            result = result @ Q
            Q = identity
            # Rebuild the residual from the accumulated transform instead of
            # continuing to propagate its low-precision Gram error.
            result_power = torch.linalg.matrix_power(result, power)
            R = A @ result_power
            R = (R + R.mT) * 0.5
    return (result @ Q) / scale.pow(1.0 / power)


def muon_polar(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Approximate ``U V.T`` using the supplied five quintic stages."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        update = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps) @ X
    else:
        gram = X.mT @ X
        update = X @ _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
    return update.to(original_dtype)


def attention_operator_muon_updates(
    query_weight: Tensor,
    key_weight: Tensor,
    query_gradient: Tensor,
    key_gradient: Tensor,
    num_heads: int,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Lift a tangent-projected Muon update of each head's ``Wq.T @ Wk``.

    PyTorch Linear weights are output-by-input.  Within each attention head we
    therefore use ``A = Wq_head.T`` and ``B = Wk_head.T``, so the bilinear
    attention operator is ``M = A @ B.T``.  The ambient gradient is recovered
    jointly from both factor gradients, polarized in M-space, projected onto
    the rank-h tangent space, and lifted back with a balanced minimum-norm
    factor update.
    """
    if query_weight.shape != key_weight.shape:
        raise ValueError("query and key weights must have identical shapes")
    if query_weight.ndim != 2 or query_weight.shape[0] % num_heads:
        raise ValueError("attention output dimension must divide into heads")

    output_size, input_size = query_weight.shape
    head_size = output_size // num_heads
    query_updates = torch.empty_like(query_weight)
    key_updates = torch.empty_like(key_weight)

    for head in range(num_heads):
        rows = slice(head * head_size, (head + 1) * head_size)
        A = query_weight[rows].mT.float()
        B = key_weight[rows].mT.float()
        grad_A = query_gradient[rows].mT.float()
        grad_B = key_gradient[rows].mT.float()
        eye_h = torch.eye(head_size, device=A.device, dtype=A.dtype)

        gram_A = A.mT @ A
        gram_B = B.mT @ B
        ridge_A = eps * gram_A.norm().clamp_min(1e-12)
        ridge_B = eps * gram_B.norm().clamp_min(1e-12)
        A_plus = torch.linalg.solve(gram_A + ridge_A * eye_h, A.mT)
        B_plus = torch.linalg.solve(gram_B + ridge_B * eye_h, B.mT)
        projector_A = A @ A_plus
        projector_B = B @ B_plus

        # Minimum-Frobenius-norm H satisfying H B = grad_A first, followed
        # by the orthogonal complement needed to satisfy A.T H = grad_B.T.
        H = grad_A @ B_plus
        residual = grad_B.mT - A.mT @ H
        H = H + A_plus.mT @ (residual - residual @ projector_B)

        polar = muon_polar(H, eps).float()
        left = projector_A @ polar
        tangent = left + polar @ projector_B - left @ projector_B

        # This symmetric half/half lift reproduces the tangent direction:
        # dA B.T + A dB.T = tangent (up to ridge regularization).
        dA = (tangent - 0.5 * projector_A @ tangent) @ B @ torch.linalg.inv(
            gram_B + ridge_B * eye_h)
        dB = (tangent.mT - 0.5 * projector_B @ tangent.mT) @ A @ torch.linalg.inv(
            gram_A + ridge_A * eye_h)
        query_updates[rows] = dA.mT.to(query_weight.dtype)
        key_updates[rows] = dB.mT.to(key_weight.dtype)

    return query_updates, key_updates


def muon_progressive_polished(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Three ordinary Muon stages followed by progressive row polishing."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        X = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps) @ X
    else:
        gram = X.mT @ X
        X = X @ _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    for fraction in (0.5, 0.25, 0.125, 0.125, 0.125):
        X = subsampled_polar_polish(
            X, fraction=fraction, steps=1, eta=0.5)
    if transposed:
        X = X.mT
    return X.to(original_dtype)


def muon_equivariant_polished(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Three ordinary Muon stages followed by two full Newton steps."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        X = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps) @ X
    else:
        gram = X.mT @ X
        X = X @ _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    for _ in range(2):
        X = 1.5 * X - 0.5 * (X @ X.mT) @ X
    if transposed:
        X = X.mT
    return X.to(original_dtype)


def _muon_after_three_stages(G: Tensor, eps: float) -> Tensor:
    """Apply exactly the first three ordinary Muon stages."""
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        return _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps) @ X
    gram = X.mT @ X
    return X @ _gram_inverse_root(
        gram, 2, INVERSE_SQRT_EXPRESS_COEFFS[:3], eps)


def muon_adaptive(
    G: Tensor,
    eps: float = 1e-7,
    stage_four_threshold: float = 0.05,
    stage_five_threshold: float = 0.01,
) -> Tensor:
    """Use three, four, or five Muon stages based on polar residual."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = _muon_after_three_stages(G, eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT

    identity = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    thresholds = (stage_four_threshold, stage_five_threshold)
    for threshold, (a, b, c) in zip(
        thresholds, INVERSE_SQRT_EXPRESS_COEFFS[3:]
    ):
        gram = X @ X.mT
        normalized_residual = (gram - identity).norm() / X.shape[0] ** 0.5
        if normalized_residual.item() <= threshold:
            break
        X = a * X + (b * gram + c * (gram @ gram)) @ X

    if transposed:
        X = X.mT
    return X.to(original_dtype)


def muon_spectral_sketch_polished(
    G: Tensor,
    eps: float = 1e-7,
    rank_fraction: float = 0.125,
    seed: int = 0,
) -> Tensor:
    """Three Muon stages plus projected Newton correction in a residual sketch."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    if not 0.0 < rank_fraction <= 1.0:
        raise ValueError("sketch rank fraction must be in (0, 1]")
    original_dtype = G.dtype
    X = _muon_after_three_stages(G, eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT

    rows = X.shape[0]
    rank = max(1, min(rows, round(rows * rank_fraction)))
    devices = [X.device] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        sketch = torch.randn(rows, rank, device=X.device, dtype=X.dtype)
    residual_sketch = X @ (X.mT @ sketch) - sketch
    basis = torch.linalg.qr(residual_sketch, mode="reduced").Q
    residual_basis = X @ (X.mT @ basis) - basis
    X = X - 0.5 * basis @ (residual_basis.mT @ X)

    if transposed:
        X = X.mT
    return X.to(original_dtype)


def muon_split_merge(
    G: Tensor,
    full_polish_steps: int,
    safety_margin: float = 1.05,
) -> Tensor:
    """Lift two row halves independently, merge, then polish globally."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    if full_polish_steps not in (1, 2):
        raise ValueError("split-merge Muon expects one or two full polish steps")
    original_dtype = G.dtype
    X = G.float()
    X = X / X.norm().clamp_min(torch.finfo(X.dtype).tiny)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT

    lifted_blocks = []
    for block in X.tensor_split(2, dim=0):
        if block.shape[0] == 0:
            continue
        for a, b, c in INVERSE_SQRT_EXPRESS_COEFFS[:3]:
            gram = block @ block.mT
            block = a * block + (b * gram + c * (gram @ gram)) @ block
        lifted_blocks.append(block)
    X = torch.cat(lifted_blocks, dim=0)

    # Each independently lifted block has operator norm at most about 1.032
    # over the design interval. The concatenation is therefore bounded by
    # sqrt(2) times that value; this fixed scale avoids a global SVD/power pass.
    X = X / (2.0 ** 0.5 * safety_margin)
    for _ in range(full_polish_steps):
        X = 1.5 * X - 0.5 * (X @ X.mT) @ X

    if transposed:
        X = X.mT
    return X.to(original_dtype)


def muon_inverse_half(G: Tensor, eps: float = 1e-4) -> Tensor:
    """Return damped ``U S**(-1/2) V.T`` using the smaller Gram matrix.

    If ``A`` is the smaller of ``G G.T`` and ``G.T G``, the update applies
    ``A**(-3/4) = A**(-1/2) A**(-1/4)`` on the corresponding side.  No SVD or
    explicit inverse is formed.
    """
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        update = (half @ quarter) @ X
    else:
        gram = X.mT @ X
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        update = X @ (half @ quarter)
    return update.to(original_dtype)


def muon_inverse_half_cubed_quarter(G: Tensor, eps: float = 1e-4) -> Tensor:
    """Return ``U S**(-1/2) V.T`` by cubing one Gram inverse fourth root."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        quarter_squared = quarter @ quarter
        inverse_three_fourths = quarter_squared @ quarter
        inverse_three_fourths = (
            inverse_three_fourths + inverse_three_fourths.mT) * 0.5
        update = inverse_three_fourths @ X
    else:
        gram = X.mT @ X
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        quarter_squared = quarter @ quarter
        inverse_three_fourths = quarter_squared @ quarter
        inverse_three_fourths = (
            inverse_three_fourths + inverse_three_fourths.mT) * 0.5
        update = X @ inverse_three_fourths
    return update.to(original_dtype)


def muon_randomized_svd_inverse_half(
    G: Tensor,
    eps: float = 1e-4,
    niter: int = 2,
    seed: int = 0,
) -> Tensor:
    """Return damped inverse-half Muon via a full-rank randomized SVD.

    ``q=min(G.shape)`` avoids truncating precisely the spectral tail under
    investigation.  The explicit seed keeps replicated DDP optimizers in
    sync even though the training script seeds data RNGs per rank.
    """
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if niter < 0:
        raise ValueError("randomized SVD power iterations must be nonnegative")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    devices = [X.device] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        U, singular, V = torch.svd_lowrank(
            X, q=min(X.shape), niter=niter)
    # Match the Gram-NS damping convention without forming a Gram matrix:
    # ||X X.T||_F = sqrt(sum_i sigma_i**4).
    damping = eps * singular.square().norm()
    weights = singular * (singular.square() + damping).pow(-0.75)
    return ((U * weights) @ V.mT).to(original_dtype)


def muon_randomized_svd_pinv(
    G: Tensor,
    eps: float = 1e-4,
    niter: int = 2,
    seed: int = 0,
) -> Tensor:
    """Return damped ``(G^+)^T = U S^-1 V.T`` via randomized SVD."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if niter < 0:
        raise ValueError("randomized SVD power iterations must be nonnegative")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    devices = [X.device] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        U, singular, V = torch.svd_lowrank(X, q=min(X.shape), niter=niter)
    damping = eps * singular.square().norm()
    weights = singular / (singular.square() + damping)
    return ((U * weights) @ V.mT).to(original_dtype)


def _gram_inverse(matrix: Tensor, eps: float, steps: int = 8) -> Tensor:
    """Cubic Newton-Schulz inverse of a relatively damped SPD matrix."""
    if steps < 1:
        raise ValueError("Gram inverse needs at least one iteration")
    scale = matrix.norm().clamp_min(torch.finfo(matrix.dtype).tiny)
    A = matrix / scale
    identity = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A = A + eps * identity
    inverse = identity
    for _ in range(steps):
        error = identity - A @ inverse
        inverse = inverse @ (identity + error + error @ error)
        inverse = (inverse + inverse.mT) * 0.5
    return inverse / scale


def muon_ns_pinv(G: Tensor, eps: float = 1e-4, steps: int = 8) -> Tensor:
    """Return damped transposed-pseudoinverse update using Gram NS."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        update = _gram_inverse(X @ X.mT, eps, steps) @ X
    else:
        update = X @ _gram_inverse(X.mT @ X, eps, steps)
    return update.to(original_dtype)


def muon_from_gram_inverse_half(
    numerator: Tensor,
    gram: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """Precondition a numerator by power -3/4 of an external Gram EMA."""
    half = _gram_inverse_root(
        gram.float(), 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
    quarter = _gram_inverse_root(
        gram.float(), 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
    preconditioner = half @ quarter
    X = numerator.float()
    update = preconditioner @ X if gram.shape[0] == X.shape[0] \
        else X @ preconditioner
    return update.to(numerator.dtype)


def muon_from_gram_pinv(
    numerator: Tensor,
    gram: Tensor,
    eps: float = 1e-4,
    steps: int = 8,
) -> Tensor:
    """Precondition a numerator by power -1 of an external Gram EMA."""
    preconditioner = _gram_inverse(gram.float(), eps, steps)
    X = numerator.float()
    update = preconditioner @ X if gram.shape[0] == X.shape[0] \
        else X @ preconditioner
    return update.to(numerator.dtype)


def shampoo_inverse_fourth(gram: Tensor, eps: float = 1e-4) -> Tensor:
    """Return the damped inverse fourth root used by SOAP-Muon interfaces."""
    return _gram_inverse_root(
        gram.float(), 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)


def generalized_muon(G: Tensor, power: float, eps: float = 1e-4) -> Tensor:
    """Return ``U S**power V.T`` for power zero or one half."""
    if power == 0.0:
        return muon_polar(G, eps)
    if power != 0.5:
        raise ValueError("generalized Muon currently supports powers 0 and 1/2")
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        update = shampoo_inverse_fourth(X @ X.mT, eps) @ X
    else:
        update = X @ shampoo_inverse_fourth(X.mT @ X, eps)
    return update.to(original_dtype)


def joint_muon_pair_updates(
    first: Tensor,
    second: Tensor,
    *,
    power: float,
    mix_lambda: float = 0.5,
    normalize_blocks: bool = True,
    eps: float = 1e-4,
    diagnostics: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Jointly transform momentum updates sharing one compatible interface."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[1]:
        raise ValueError("joint Muon tensors must share first.rows == second.columns")
    if not 0.0 <= mix_lambda <= 1.0:
        raise ValueError("joint Muon lambda must be in [0, 1]")
    first_block = first.float()
    second_block = second.mT.float()
    if normalize_blocks:
        first_block = first_block / first_block.norm().clamp_min(1e-12)
        second_block = second_block / second_block.norm().clamp_min(1e-12)
    first_block = first_block * mix_lambda ** 0.5
    second_block = second_block * (1.0 - mix_lambda) ** 0.5
    joined = torch.cat((first_block, second_block), dim=1)
    transformed = generalized_muon(joined, power, eps).float()
    first_width = first.shape[1]
    if diagnostics is not None:
        singular = torch.linalg.svdvals(joined)
        transformed_singular = torch.linalg.svdvals(transformed)
        diagnostics["input_sigma_min"] = singular.min()
        diagnostics["input_sigma_max"] = singular.max()
        diagnostics["output_sigma_min"] = transformed_singular.min()
        diagnostics["output_sigma_max"] = transformed_singular.max()
        diagnostics["n_nonfinite"] = (~torch.isfinite(transformed)).sum()
        first_energy = transformed[:, :first_width].square().sum()
        total_energy = transformed.square().sum().clamp_min(1e-30)
        diagnostics["first_energy_fraction"] = first_energy / total_energy
        if power == 0.5:
            gram = joined @ joined.mT
            damping = eps * gram.norm()
            eigenvalues, eigenvectors = torch.linalg.eigh(
                (gram + gram.mT) * 0.5)
            exact_root = (
                eigenvectors
                * (eigenvalues.clamp_min(0) + damping).pow(-0.25)
            ) @ eigenvectors.mT
            exact = exact_root @ joined
            diagnostics["ns_relative_error"] = (
                (transformed - exact).norm() / exact.norm().clamp_min(1e-30))
    return (transformed[:, :first_width].to(first.dtype),
            transformed[:, first_width:].mT.to(second.dtype))


def muon_projected_inverse_half(
    G: Tensor,
    eps: float = 1e-4,
    beta: float = 0.25,
) -> Tensor:
    """Shrink inverse-half spectral reweighting toward ordinary Muon."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("projection beta must be in [0, 1]")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = half @ X
        inverse_half = (half @ quarter) @ X
    else:
        gram = X.mT @ X
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = X @ half
        inverse_half = X @ (half @ quarter)
    coefficient = (inverse_half * polar).sum() / polar.square().sum().clamp_min(1e-12)
    parallel = polar * coefficient
    return (parallel + beta * (inverse_half - parallel)).to(original_dtype)


def muon_gated_inverse_half(G: Tensor, eps: float = 1e-4) -> Tensor:
    """Gate inverse-half modes by the squared soft-polar response.

    Computes ``Q @ (Q.T @ H)`` associatively through the smaller dimension,
    where ``Q`` is damped Muon and ``H`` is damped inverse-half Muon.
    """
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = half @ X
        inverse_half = (half @ quarter) @ X
        update = (polar @ polar.mT) @ inverse_half
    else:
        gram = X.mT @ X
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = X @ half
        inverse_half = X @ (half @ quarter)
        update = polar @ (polar.mT @ inverse_half)
    return update.to(original_dtype)


def muon_correlation_gated_inverse_half(G: Tensor, eps: float = 1e-4) -> Tensor:
    """Apply the soft-polar gate as ``Q @ (Q.T @ H)`` in that order."""
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}")
    if G.numel() == 0:
        return G.clone()
    original_dtype = G.dtype
    X = G.float()
    if X.shape[0] <= X.shape[1]:
        gram = X @ X.mT
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = half @ X
        inverse_half = (half @ quarter) @ X
    else:
        gram = X.mT @ X
        half = _gram_inverse_root(
            gram, 2, INVERSE_SQRT_EXPRESS_COEFFS, eps)
        quarter = _gram_inverse_root(
            gram, 4, INVERSE_FOURTH_EXPRESS_COEFFS, eps)
        polar = X @ half
        inverse_half = X @ (half @ quarter)
    return (polar @ (polar.mT @ inverse_half)).to(original_dtype)


class MuonWithAuxAdam(Optimizer):
    """Muon for selected matrices and AdamW for every other parameter."""

    def __init__(
        self,
        muon_params,
        adam_params,
        *,
        lr: float,
        adam_lr: float,
        variant: str = "polar",
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        spectral_eps: float = 1e-4,
        rms_scale: float | None = None,
        projection_beta: float = 0.25,
        randomized_svd_niter: int = 2,
        randomized_svd_seed: int = 17,
        pinv_ns_steps: int = 8,
        gram_beta2: float = 0.99,
        interface_mix_lambda: float = 0.5,
        joint_mix_lambda: float = 0.5,
        joint_normalize_blocks: bool = True,
        joint_diagnostics_every: int = 0,
        spectral_diagnostics_every: int = 0,
        pre_ns_sphere_strength: float = 0.03,
        pre_ns_sphere_log_error_clip: float = 0.5,
        attention_pairs=(),
    ):
        if variant not in ("polar", "stable", "polished", "progressive-polished",
                           "equivariant-polished", "inverse-half", "projected-inverse-half",
                           "adaptive", "spectral-sketch-polished",
                           "split-merge-one", "split-merge-two",
                           "inverse-half-cubed-quarter",
                           "gated-inverse-half", "correlation-gated-inverse-half",
                           "randomized-svd-inverse-half", "randomized-svd-pinv",
                           "ns-pinv", "ema-gram-inverse-half", "ema-gram-pinv",
                           "soap-muon-eigh", "soap-muon-independent",
                           "soap-muon-interface-shared", "joint-muon-p0",
                           "joint-muon-p05", "attention-operator",
                           "pre-ns-cautious-sphere"):
            raise ValueError(
                "variant must be 'polar', 'inverse-half', or "
                "'projected-inverse-half', 'gated-inverse-half', or "
                "'correlation-gated-inverse-half', or "
                "'randomized-svd-inverse-half', 'randomized-svd-pinv', or "
                "'ns-pinv', 'ema-gram-inverse-half', 'ema-gram-pinv', or "
                "'soap-muon-eigh'")
        if not 0.0 <= gram_beta2 < 1.0:
            raise ValueError("Gram EMA beta2 must be in [0, 1)")
        if not 0.0 <= interface_mix_lambda <= 1.0:
            raise ValueError("interface mix lambda must be in [0, 1]")
        if not 0.0 <= joint_mix_lambda <= 1.0:
            raise ValueError("joint Muon lambda must be in [0, 1]")
        if joint_diagnostics_every < 0:
            raise ValueError("joint diagnostics interval must be nonnegative")
        if spectral_diagnostics_every < 0:
            raise ValueError("spectral diagnostics interval must be nonnegative")
        if pre_ns_sphere_strength < 0.0:
            raise ValueError("pre-NS sphere strength must be nonnegative")
        if pre_ns_sphere_log_error_clip <= 0.0:
            raise ValueError("pre-NS sphere log-error clip must be positive")
        muon_params, adam_params = list(muon_params), list(adam_params)
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "lr": lr, "use_muon": True})
        if adam_params:
            groups.append({"params": adam_params, "lr": adam_lr, "use_muon": False})
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        variant=variant, spectral_eps=spectral_eps,
                        rms_scale=rms_scale, projection_beta=projection_beta,
                        randomized_svd_niter=randomized_svd_niter,
                        randomized_svd_seed=randomized_svd_seed,
                        pinv_ns_steps=pinv_ns_steps,
                        gram_beta2=gram_beta2,
                        interface_mix_lambda=interface_mix_lambda,
                        joint_mix_lambda=joint_mix_lambda,
                        joint_normalize_blocks=joint_normalize_blocks,
                        joint_diagnostics_every=joint_diagnostics_every,
                        spectral_diagnostics_every=spectral_diagnostics_every,
                        pre_ns_sphere_strength=pre_ns_sphere_strength,
                        pre_ns_sphere_log_error_clip=pre_ns_sphere_log_error_clip,
                        betas=(0.9, 0.999), adam_eps=1e-8)
        super().__init__(groups, defaults)
        self.attention_pairs = tuple(attention_pairs)
        self.attention_parameter_ids = {
            id(parameter)
            for query, key, _ in self.attention_pairs
            for parameter in (query, key)
        }
        self.soap_interface_grams = {}
        self.last_joint_diagnostics = {}
        self.last_spectral_diagnostics = {}
        self.last_pre_ns_diagnostics = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        spectral_rows = []
        spectral_step = 0
        pre_ns_rows = []
        for group in self.param_groups:
            if not group["use_muon"]:
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad
                    state = self.state[parameter]
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    avg, avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    correction1 = 1.0 - beta1 ** state["step"]
                    correction2 = 1.0 - beta2 ** state["step"]
                    denom = avg_sq.sqrt().div_(correction2 ** 0.5).add_(group["adam_eps"])
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(avg, denom,
                                       value=-group["lr"] / correction1)
                continue
            if group["variant"] == "attention-operator":
                for query, key, num_heads in self.attention_pairs:
                    if query.grad is None or key.grad is None:
                        continue
                    pair_state = self.state[query]
                    if "attention_query_momentum" not in pair_state:
                        pair_state["attention_query_momentum"] = torch.zeros_like(query)
                        pair_state["attention_key_momentum"] = torch.zeros_like(key)
                    query_buffer = pair_state["attention_query_momentum"]
                    key_buffer = pair_state["attention_key_momentum"]
                    query_buffer.mul_(group["momentum"]).add_(query.grad)
                    key_buffer.mul_(group["momentum"]).add_(key.grad)
                    query_gradient = query.grad.add(
                        query_buffer, alpha=group["momentum"])
                    key_gradient = key.grad.add(
                        key_buffer, alpha=group["momentum"])
                    query_update, key_update = attention_operator_muon_updates(
                        query, key, query_gradient, key_gradient, num_heads,
                        group["spectral_eps"])
                    if group["rms_scale"] is not None:
                        target = group["rms_scale"] * (
                            query.numel() + key.numel()) ** 0.5
                        norm = (query_update.square().sum()
                                + key_update.square().sum()).sqrt().clamp_min(1e-12)
                        query_update.mul_(target / norm)
                        key_update.mul_(target / norm)
                    query.mul_(1.0 - group["lr"] * group["weight_decay"])
                    key.mul_(1.0 - group["lr"] * group["weight_decay"])
                    query.add_(query_update, alpha=-group["lr"])
                    key.add_(key_update, alpha=-group["lr"])

            prepared_joint_updates = {}
            if group["variant"] in ("joint-muon-p0", "joint-muon-p05"):
                matrices = group["params"]
                momentum_updates = {}
                candidates = {}
                for parameter in matrices:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter.grad)
                        state["step"] = 0
                    state["step"] += 1
                    buffer = state["momentum_buffer"]
                    buffer.mul_(group["momentum"]).add_(parameter.grad)
                    momentum_updates[id(parameter)] = parameter.grad.add(
                        buffer, alpha=group["momentum"])
                power = 0.0 if group["variant"] == "joint-muon-p0" else 0.5
                current_step = max(
                    (self.state[p].get("step", 0) for p in matrices), default=0)
                collect_diagnostics = (
                    group["joint_diagnostics_every"] > 0
                    and current_step % group["joint_diagnostics_every"] == 0)
                diagnostic_rows = []
                for index in range(len(matrices) - 1):
                    first, second = matrices[index], matrices[index + 1]
                    if (id(first) not in momentum_updates
                            or id(second) not in momentum_updates
                            or first.shape[0] != second.shape[1]):
                        continue
                    pair_diagnostics = {} if collect_diagnostics else None
                    first_update, second_update = joint_muon_pair_updates(
                        momentum_updates[id(first)], momentum_updates[id(second)],
                        power=power, mix_lambda=group["joint_mix_lambda"],
                        normalize_blocks=group["joint_normalize_blocks"],
                        eps=group["spectral_eps"], diagnostics=pair_diagnostics)
                    if group["rms_scale"] is not None:
                        target_pair_norm = group["rms_scale"] * (
                            first_update.numel() + second_update.numel()) ** 0.5
                    else:
                        first_baseline = muon_polar(
                            momentum_updates[id(first)], group["spectral_eps"])
                        second_baseline = muon_polar(
                            momentum_updates[id(second)], group["spectral_eps"])
                        first_baseline.mul_(max(
                            1.0, first.shape[0] / first.shape[1]) ** 0.5)
                        second_baseline.mul_(max(
                            1.0, second.shape[0] / second.shape[1]) ** 0.5)
                        target_pair_norm = (
                            first_baseline.square().sum()
                            + second_baseline.square().sum()).sqrt()
                    pair_norm = (
                        first_update.square().sum()
                        + second_update.square().sum()).sqrt().clamp_min(1e-12)
                    pair_scale = target_pair_norm / pair_norm
                    if pair_diagnostics is not None:
                        pair_diagnostics["first_scale_amplification"] = pair_scale
                        pair_diagnostics["second_scale_amplification"] = pair_scale
                    for parameter, candidate in (
                        (first, first_update), (second, second_update)
                    ):
                        candidate.mul_(pair_scale)
                        candidates.setdefault(id(parameter), []).append(candidate)
                    if pair_diagnostics is not None:
                        diagnostic_rows.append(pair_diagnostics)
                for parameter in matrices:
                    parameter_candidates = candidates.get(id(parameter))
                    if parameter_candidates:
                        prepared_joint_updates[id(parameter)] = torch.stack(
                            parameter_candidates).mean(dim=0)
                    elif id(parameter) in momentum_updates:
                        fallback = muon_polar(
                            momentum_updates[id(parameter)], group["spectral_eps"])
                        if group["rms_scale"] is not None:
                            fallback.mul_(
                                group["rms_scale"] * fallback.numel() ** 0.5
                                / fallback.norm().clamp_min(1e-12))
                        else:
                            fallback.mul_(max(
                                1.0, parameter.shape[0] / parameter.shape[1]
                            ) ** 0.5)
                        prepared_joint_updates[id(parameter)] = fallback
                if diagnostic_rows:
                    self.last_joint_diagnostics = {
                        "step": current_step,
                        "pairs": len(diagnostic_rows),
                        "input_sigma_min": min(
                            row["input_sigma_min"] for row in diagnostic_rows).item(),
                        "input_sigma_max": max(
                            row["input_sigma_max"] for row in diagnostic_rows).item(),
                        "output_sigma_min": min(
                            row["output_sigma_min"] for row in diagnostic_rows).item(),
                        "output_sigma_max": max(
                            row["output_sigma_max"] for row in diagnostic_rows).item(),
                        "ns_relative_error_max": max(
                            row.get("ns_relative_error", torch.tensor(0.0))
                            for row in diagnostic_rows).item(),
                        "nonfinite": sum(
                            row["n_nonfinite"].item() for row in diagnostic_rows),
                        "first_energy_fraction_min": min(
                            row["first_energy_fraction"] for row in diagnostic_rows).item(),
                        "first_energy_fraction_max": max(
                            row["first_energy_fraction"] for row in diagnostic_rows).item(),
                        "scale_amplification_max": max(
                            max(row["first_scale_amplification"],
                                row["second_scale_amplification"])
                            for row in diagnostic_rows).item(),
                    }

            prepared_soap_updates = {}
            soap_factor_keys = {}
            if group["variant"] in (
                "soap-muon-independent", "soap-muon-interface-shared"
            ):
                matrices = group["params"]
                for index, parameter in enumerate(matrices):
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter.grad)
                        state["step"] = 0
                    state["step"] += 1
                    buffer = state["momentum_buffer"]
                    buffer.mul_(group["momentum"]).add_(parameter.grad)
                    prepared_soap_updates[id(parameter)] = parameter.grad.add(
                        buffer, alpha=group["momentum"])

                    left_key = (index, "left")
                    right_key = (index, "right")
                    if (group["variant"] == "soap-muon-interface-shared"
                            and index + 1 < len(matrices)
                            and parameter.shape[0] == matrices[index + 1].shape[1]):
                        left_key = (index, "shared")
                    if (group["variant"] == "soap-muon-interface-shared"
                            and index > 0
                            and matrices[index - 1].shape[0] == parameter.shape[1]):
                        right_key = (index - 1, "shared")
                    soap_factor_keys[id(parameter)] = (left_key, right_key)

                contributions = {}
                for parameter in matrices:
                    if parameter.grad is None:
                        continue
                    left_key, right_key = soap_factor_keys[id(parameter)]
                    gradient = parameter.grad.float()
                    for key, contribution, mix_weight in (
                        (left_key, gradient @ gradient.mT,
                         group["interface_mix_lambda"]),
                        (right_key, gradient.mT @ gradient,
                         1.0 - group["interface_mix_lambda"]),
                    ):
                        if key[1] == "shared":
                            mean_eigenvalue = contribution.trace() / contribution.shape[0]
                            contribution = contribution / (
                                mean_eigenvalue
                                + torch.finfo(contribution.dtype).eps)
                            contribution = contribution * mix_weight
                        if key in contributions:
                            contributions[key].add_(contribution)
                        else:
                            contributions[key] = contribution
                for key, contribution in contributions.items():
                    if key not in self.soap_interface_grams:
                        self.soap_interface_grams[key] = torch.zeros_like(contribution)
                    self.soap_interface_grams[key].mul_(group["gram_beta2"]).add_(
                        contribution, alpha=1.0 - group["gram_beta2"])

            for parameter_index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue
                if (group["variant"] == "attention-operator"
                        and id(parameter) in self.attention_parameter_ids):
                    continue
                G = parameter.grad
                if G.ndim != 2:
                    raise ValueError("Muon parameter routing must contain only matrices")
                state = self.state[parameter]
                if id(parameter) in prepared_joint_updates:
                    update = prepared_joint_updates[id(parameter)]
                elif id(parameter) in prepared_soap_updates:
                    update = prepared_soap_updates[id(parameter)]
                else:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(G)
                        state["step"] = 0
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(G)
                    update = G.add(buf, alpha=group["momentum"])
                if group["variant"] == "pre-ns-cautious-sphere":
                    weight = parameter.float()
                    update_float = update.float()
                    weight_norm = weight.norm().clamp_min(1e-30)
                    update_norm = update_float.norm().clamp_min(1e-30)
                    if "sphere_reference_norm" not in state:
                        state["sphere_reference_norm"] = weight_norm.item()
                    reference = state["sphere_reference_norm"]
                    log_error = torch.log(weight_norm / reference).clamp(
                        -group["pre_ns_sphere_log_error_clip"],
                        group["pre_ns_sphere_log_error_clip"],
                    )
                    alignment = log_error * (weight * update_float).sum()
                    gate = alignment >= 0
                    bias_scale = (
                        group["pre_ns_sphere_strength"] * log_error
                        * update_norm / weight_norm
                    )
                    bias = weight * bias_scale if gate else torch.zeros_like(weight)
                    update = (update_float + bias).to(update.dtype)
                    pre_ns_rows.append({
                        "step": state["step"],
                        "abs_log_radius_error": log_error.abs().item(),
                        "gate": float(gate.item()),
                        "bias_to_momentum": (
                            bias.norm() / update_norm).item(),
                    })
                if group["variant"] == "soap-muon-eigh":
                    update_float = update.float()
                    if "soap_exp_avg_sq" not in state:
                        rows, columns = G.shape
                        state["soap_exp_avg_sq"] = torch.zeros_like(update_float)
                        state["soap_row_gram"] = torch.zeros(
                            rows, rows, device=G.device, dtype=torch.float32)
                        state["soap_col_gram"] = torch.zeros(
                            columns, columns, device=G.device, dtype=torch.float32)
                        state["soap_row_basis"] = None
                        state["soap_col_basis"] = None
                    row_basis = state["soap_row_basis"]
                    col_basis = state["soap_col_basis"]
                    if row_basis is not None:
                        projected = row_basis.mT @ update_float @ col_basis
                        second_moment = state["soap_exp_avg_sq"]
                        second_moment.mul_(group["gram_beta2"]).addcmul_(
                            projected, projected,
                            value=1.0 - group["gram_beta2"])
                        preconditioned = row_basis @ (
                            projected / second_moment.clamp_min(1e-16).sqrt()
                        ) @ col_basis.mT
                        preconditioned.mul_(
                            update_float.norm()
                            / preconditioned.norm().clamp_min(1e-12))
                        update = preconditioned.to(update.dtype)
                    update = muon_polar(update, group["spectral_eps"])
                    gradient_float = G.float()
                    state["soap_row_gram"].lerp_(
                        gradient_float @ gradient_float.mT,
                        1.0 - group["gram_beta2"])
                    state["soap_col_gram"].lerp_(
                        gradient_float.mT @ gradient_float,
                        1.0 - group["gram_beta2"])
                    state["soap_row_basis"] = torch.linalg.eigh(
                        (state["soap_row_gram"] + state["soap_row_gram"].mT)
                        * 0.5)[1].flip(1)
                    state["soap_col_basis"] = torch.linalg.eigh(
                        (state["soap_col_gram"] + state["soap_col_gram"].mT)
                        * 0.5)[1].flip(1)
                elif group["variant"] in (
                    "soap-muon-independent", "soap-muon-interface-shared"
                ):
                    left_key, right_key = soap_factor_keys[id(parameter)]
                    left = shampoo_inverse_fourth(
                        self.soap_interface_grams[left_key],
                        group["spectral_eps"])
                    right = shampoo_inverse_fourth(
                        self.soap_interface_grams[right_key],
                        group["spectral_eps"])
                    update = left @ update.float() @ right
                    update = muon_polar(update, group["spectral_eps"])
                elif group["variant"] in ("joint-muon-p0", "joint-muon-p05"):
                    pass
                elif group["variant"] in (
                    "ema-gram-inverse-half", "ema-gram-pinv"
                ):
                    raw_gram = G @ G.mT if G.shape[0] <= G.shape[1] else G.mT @ G
                    if "gram_ema" not in state:
                        state["gram_ema"] = torch.zeros_like(raw_gram)
                    gram_ema = state["gram_ema"]
                    gram_ema.mul_(group["gram_beta2"]).add_(
                        raw_gram, alpha=1.0 - group["gram_beta2"])
                    if group["variant"] == "ema-gram-inverse-half":
                        update = muon_from_gram_inverse_half(
                            update, gram_ema, group["spectral_eps"])
                    else:
                        update = muon_from_gram_pinv(
                            update, gram_ema, group["spectral_eps"],
                            group["pinv_ns_steps"])
                elif group["variant"] in (
                    "polar", "attention-operator", "pre-ns-cautious-sphere"
                ):
                    update = muon_polar(update, group["spectral_eps"])
                elif group["variant"] == "stable":
                    update = muon_stable(update)
                elif group["variant"] == "polished":
                    update = muon_polished(update, group["spectral_eps"])
                elif group["variant"] == "progressive-polished":
                    update = muon_progressive_polished(
                        update, group["spectral_eps"])
                elif group["variant"] == "equivariant-polished":
                    update = muon_equivariant_polished(
                        update, group["spectral_eps"])
                elif group["variant"] == "adaptive":
                    update = muon_adaptive(update, group["spectral_eps"])
                elif group["variant"] == "spectral-sketch-polished":
                    sketch_seed = (
                        group["randomized_svd_seed"]
                        + state["step"] * len(group["params"])
                        + parameter_index
                    )
                    update = muon_spectral_sketch_polished(
                        update, group["spectral_eps"], seed=sketch_seed)
                elif group["variant"] in ("split-merge-one", "split-merge-two"):
                    update = muon_split_merge(
                        update,
                        full_polish_steps=(
                            1 if group["variant"] == "split-merge-one" else 2
                        ),
                    )
                elif group["variant"] == "inverse-half":
                    spectral_input = update.float()
                    update = muon_inverse_half(update, group["spectral_eps"])
                    spectral_step = max(spectral_step, state["step"])
                    if (group["spectral_diagnostics_every"] > 0
                            and state["step"] % group["spectral_diagnostics_every"] == 0):
                        X = spectral_input
                        left_side = X.shape[0] <= X.shape[1]
                        gram = X @ X.mT if left_side else X.mT @ X
                        gram = (gram + gram.mT) * 0.5
                        damping = group["spectral_eps"] * gram.norm()
                        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
                        exact_root = (
                            eigenvectors
                            * (eigenvalues.clamp_min(0) + damping).pow(-0.75)
                        ) @ eigenvectors.mT
                        exact = exact_root @ X if left_side else X @ exact_root
                        input_singular = eigenvalues.clamp_min(0).sqrt()
                        output_singular = torch.linalg.svdvals(update.float())
                        spectral_rows.append({
                            "relative_error": (
                                (update.float() - exact).norm()
                                / exact.norm().clamp_min(1e-30)),
                            "input_sigma_min": input_singular.min(),
                            "input_sigma_max": input_singular.max(),
                            "output_sigma_min": output_singular.min(),
                            "output_sigma_max": output_singular.max(),
                            "raw_output_norm": update.float().norm(),
                            "nonfinite": (~torch.isfinite(update)).sum(),
                        })
                elif group["variant"] == "inverse-half-cubed-quarter":
                    update = muon_inverse_half_cubed_quarter(
                        update, group["spectral_eps"])
                elif group["variant"] == "projected-inverse-half":
                    update = muon_projected_inverse_half(
                        update, group["spectral_eps"], group["projection_beta"])
                elif group["variant"] == "gated-inverse-half":
                    update = muon_gated_inverse_half(
                        update, group["spectral_eps"])
                elif group["variant"] == "correlation-gated-inverse-half":
                    update = muon_correlation_gated_inverse_half(
                        update, group["spectral_eps"])
                elif group["variant"] in (
                    "randomized-svd-inverse-half", "randomized-svd-pinv"
                ):
                    randomized_seed = (
                        group["randomized_svd_seed"]
                        + state["step"] * len(group["params"])
                        + parameter_index
                    )
                    randomized_function = (
                        muon_randomized_svd_inverse_half
                        if group["variant"] == "randomized-svd-inverse-half"
                        else muon_randomized_svd_pinv
                    )
                    update = randomized_function(
                        update, group["spectral_eps"],
                        group["randomized_svd_niter"], randomized_seed)
                else:
                    update = muon_ns_pinv(
                        update, group["spectral_eps"], group["pinv_ns_steps"])
                if group["variant"] in ("joint-muon-p0", "joint-muon-p05"):
                    # Joint candidates already receive one common pair scale.
                    # A second per-tensor normalization would destroy the
                    # relative energy allocation produced by the joint map.
                    pass
                elif group["rms_scale"] is not None:
                    # Kimi-style Adam RMS matching: the final matrix update
                    # has elementwise RMS exactly rms_scale, independent of
                    # its spectral power or aspect ratio.
                    target_norm = group["rms_scale"] * update.numel() ** 0.5
                    if (group["variant"] == "inverse-half"
                            and spectral_rows
                            and state["step"] % group["spectral_diagnostics_every"] == 0):
                        spectral_rows[-1]["scale_amplification"] = (
                            target_norm / update.norm().clamp_min(1e-12))
                    update.mul_(target_norm / update.norm().clamp_min(1e-12))
                else:
                    # Match the standard Muon rectangular-matrix adjustment.
                    update.mul_(max(1.0, G.shape[0] / G.shape[1]) ** 0.5)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        if spectral_rows:
            self.last_spectral_diagnostics = {
                "step": spectral_step,
                "matrices": len(spectral_rows),
                "relative_error_max": max(
                    row["relative_error"] for row in spectral_rows).item(),
                "input_sigma_min": min(
                    row["input_sigma_min"] for row in spectral_rows).item(),
                "input_sigma_max": max(
                    row["input_sigma_max"] for row in spectral_rows).item(),
                "output_sigma_min": min(
                    row["output_sigma_min"] for row in spectral_rows).item(),
                "output_sigma_max": max(
                    row["output_sigma_max"] for row in spectral_rows).item(),
                "scale_amplification_max": max(
                    row.get("scale_amplification", torch.tensor(1.0))
                    for row in spectral_rows).item(),
                "raw_output_norm_min": min(
                    row["raw_output_norm"] for row in spectral_rows).item(),
                "nonfinite": sum(row["nonfinite"].item() for row in spectral_rows),
            }
        if pre_ns_rows:
            self.last_pre_ns_diagnostics = {
                "step": max(row["step"] for row in pre_ns_rows),
                "matrix_count": len(pre_ns_rows),
                "abs_log_radius_error_mean": sum(
                    row["abs_log_radius_error"] for row in pre_ns_rows
                ) / len(pre_ns_rows),
                "gate_fraction": sum(row["gate"] for row in pre_ns_rows)
                                 / len(pre_ns_rows),
                "bias_to_momentum_mean": sum(
                    row["bias_to_momentum"] for row in pre_ns_rows
                ) / len(pre_ns_rows),
            }
        return loss

