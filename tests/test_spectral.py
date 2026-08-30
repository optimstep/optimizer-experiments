import torch

from optimizers.spectral import inverse_half_muon, newton_schulz_polar


def test_polar_has_unit_nonzero_singular_values():
    torch.manual_seed(0)
    gradient = torch.randn(16, 32)
    update = newton_schulz_polar(gradient)
    singular_values = torch.linalg.svdvals(update)
    assert torch.isfinite(update).all()
    assert (singular_values - 1).abs().mean() < 0.12


def test_inverse_half_maps_singular_values_to_square_roots():
    torch.manual_seed(1)
    gradient = torch.randn(12, 20)
    update = inverse_half_muon(gradient, epsilon=1e-8)
    expected = torch.linalg.svdvals(gradient).sqrt()
    actual = torch.linalg.svdvals(update)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

