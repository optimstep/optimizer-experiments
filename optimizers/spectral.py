"""Small, independently testable spectral transforms used by Muon variants."""
import torch


POLAR_COEFFICIENTS = (
    (3.7735, -8.3649, 4.7397),
    (4.1015, -10.5015, 6.8262),
    (3.4766, -8.7202, 5.8348),
    (2.7722, -7.0218, 5.7043),
    (3.2507, -6.2916, 5.2491),
)


def newton_schulz_polar(gradient, coefficients=POLAR_COEFFICIENTS):
    transposed = gradient.shape[-2] > gradient.shape[-1]
    x = gradient.mT if transposed else gradient
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for a, b, c in coefficients:
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


def inverse_half_muon(gradient, epsilon=1e-3):
    """Reference generalized Muon U sqrt(S) V^T for numerical validation."""
    u, s, vh = torch.linalg.svd(gradient.float(), full_matrices=False)
    floor = epsilon * s.amax(dim=-1, keepdim=True)
    return ((u * s.clamp_min(floor).sqrt().unsqueeze(-2)) @ vh).to(gradient.dtype)

