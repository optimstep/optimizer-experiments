import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from muon import (INVERSE_FOURTH_EXPRESS_COEFFS, MuonWithAuxAdam,
                  attention_operator_muon_updates,
                  generalized_muon, joint_muon_pair_updates,
                  muon_adaptive,
                  muon_correlation_gated_inverse_half,
                  muon_gated_inverse_half, muon_inverse_half, muon_polar,
                  muon_inverse_half_cubed_quarter,
                  muon_from_gram_inverse_half, muon_from_gram_pinv,
                  muon_ns_pinv,
                  muon_projected_inverse_half,
                  muon_randomized_svd_inverse_half,
                  muon_randomized_svd_pinv, muon_polished,
                  muon_equivariant_polished, muon_progressive_polished,
                  muon_spectral_sketch_polished,
                  muon_split_merge,
                  subsampled_polar_polish)


def polar_residual(matrix):
    wide = matrix if matrix.shape[0] <= matrix.shape[1] else matrix.mT
    error = wide @ wide.mT - torch.eye(wide.shape[0], dtype=wide.dtype)
    return error.norm()


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7), (1, 9)])
def test_polished_muon_is_finite_and_reduces_polar_residual(shape):
    torch.manual_seed(12)
    gradient = torch.randn(shape)
    unpolished = muon_polished(gradient, steps=0)
    polished = muon_polished(gradient)
    assert torch.isfinite(polished).all()
    residual_before = polar_residual(unpolished)
    residual_after = polar_residual(polished)
    assert residual_after <= residual_before + 2 * torch.finfo(torch.float32).eps
    if min(shape) > 1:
        assert residual_after < residual_before


def test_full_fraction_polish_matches_full_newton_polar_step():
    torch.manual_seed(13)
    matrix = torch.randn(5, 8)
    matrix /= matrix.norm()
    expected = 1.5 * matrix - 0.5 * (matrix @ matrix.mT) @ matrix
    actual = subsampled_polar_polish(matrix, fraction=1.0, steps=1)
    torch.testing.assert_close(actual, expected)


def test_disabled_polish_returns_input_without_copying():
    matrix = torch.randn(5, 8)
    assert subsampled_polar_polish(matrix, steps=0) is matrix


def test_polished_muon_without_polishing_is_exactly_ordinary_muon():
    torch.manual_seed(14)
    gradient = torch.randn(8, 5)
    epsilon = 1e-3
    torch.testing.assert_close(muon_polar(gradient, epsilon),
                               muon_polished(gradient, epsilon, steps=0),
                               rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_progressive_polishing_is_finite_and_nearly_polar(shape):
    torch.manual_seed(15)
    gradient = torch.randn(shape)
    update = muon_progressive_polished(gradient, eps=1e-3)
    assert torch.isfinite(update).all()
    assert polar_residual(update.float()) < 0.2


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_equivariant_polishing_is_finite_and_nearly_polar(shape):
    torch.manual_seed(16)
    gradient = torch.randn(shape)
    update = muon_equivariant_polished(gradient, eps=1e-3)
    assert torch.isfinite(update).all()
    assert polar_residual(update.float()) < 0.2


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_adaptive_muon_is_finite_and_nearly_polar(shape):
    torch.manual_seed(17)
    gradient = torch.randn(shape)
    update = muon_adaptive(gradient, eps=1e-3)
    assert torch.isfinite(update).all()
    assert polar_residual(update.float()) < 0.2


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_spectral_sketch_polishing_is_deterministic_and_finite(shape):
    torch.manual_seed(18)
    gradient = torch.randn(shape)
    first = muon_spectral_sketch_polished(gradient, eps=1e-3, seed=123)
    second = muon_spectral_sketch_polished(gradient, eps=1e-3, seed=123)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7), (1, 9)])
@pytest.mark.parametrize("full_steps", [1, 2])
def test_split_merge_muon_is_finite(shape, full_steps):
    torch.manual_seed(19)
    update = muon_split_merge(torch.randn(shape), full_steps)
    assert torch.isfinite(update).all()
    assert update.shape == shape


def test_inverse_fourth_schedule_on_design_interval():
    eigenvalue = torch.logspace(-3, torch.log10(torch.tensor(1.05)).item(),
                               100_000, dtype=torch.float64)
    estimate = torch.ones_like(eigenvalue)
    for a, b, c in INVERSE_FOURTH_EXPRESS_COEFFS:
        residual = eigenvalue * estimate.pow(4)
        estimate *= a + b * residual + c * residual.square()
    relative_error = (estimate * eigenvalue.pow(0.25) - 1).abs().max()
    assert relative_error < 1e-12


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_muon_spectral_updates_match_damped_svd(shape):
    torch.manual_seed(0)
    gradient = torch.randn(shape)
    epsilon = 1e-3
    gram = (gradient @ gradient.mT if shape[0] <= shape[1]
            else gradient.mT @ gradient)
    damping = epsilon * gram.norm()
    u, singular, vh = torch.linalg.svd(gradient, full_matrices=False)
    damped_sq = singular.square() + damping

    polar_ref = (u * (singular * damped_sq.rsqrt())) @ vh
    inverse_half_ref = (u * (singular * damped_sq.pow(-0.75))) @ vh

    torch.testing.assert_close(muon_polar(gradient, epsilon), polar_ref,
                               rtol=2e-4, atol=2e-5)
    inverse_half = muon_inverse_half(gradient, epsilon)
    cubed_quarter = muon_inverse_half_cubed_quarter(gradient, epsilon)
    # Relative elementwise error is ill-defined where the dense SVD reference
    # cancels to almost zero; use the operator's natural normwise error.
    assert (inverse_half - inverse_half_ref).norm() / inverse_half_ref.norm() < 3e-4
    assert (inverse_half - inverse_half_ref).abs().max() < 1.5e-4
    assert (cubed_quarter - inverse_half_ref).norm() / inverse_half_ref.norm() < 5e-4
    torch.testing.assert_close(cubed_quarter, inverse_half,
                               rtol=6e-4, atol=2e-4)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_randomized_svd_inverse_half_matches_full_svd(shape):
    torch.manual_seed(6)
    gradient = torch.randn(shape)
    epsilon = 1e-3
    u, singular, vh = torch.linalg.svd(gradient, full_matrices=False)
    damping = epsilon * singular.square().norm()
    expected = (u * (singular * (singular.square() + damping).pow(-0.75))) @ vh
    actual = muon_randomized_svd_inverse_half(
        gradient, epsilon, niter=2, seed=123)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    repeated = muon_randomized_svd_inverse_half(
        gradient, epsilon, niter=2, seed=123)
    torch.testing.assert_close(actual, repeated, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (7, 7)])
def test_pinv_backends_match_damped_full_svd(shape):
    torch.manual_seed(7)
    gradient = torch.randn(shape)
    epsilon = 1e-3
    u, singular, vh = torch.linalg.svd(gradient, full_matrices=False)
    damping = epsilon * singular.square().norm()
    expected = (u * (singular / (singular.square() + damping))) @ vh
    randomized = muon_randomized_svd_pinv(
        gradient, epsilon, niter=2, seed=321)
    ns = muon_ns_pinv(gradient, epsilon, steps=8)
    torch.testing.assert_close(randomized, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(ns, expected, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5)])
def test_external_gram_backends_match_eigendecomposition(shape):
    torch.manual_seed(8)
    numerator = torch.randn(shape)
    sample = torch.randn(shape)
    gram = sample @ sample.mT if shape[0] <= shape[1] else sample.mT @ sample
    epsilon = 1e-3
    scale = gram.norm()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram + epsilon * scale * torch.eye(gram.shape[0]))
    inverse_half_preconditioner = (
        eigenvectors * eigenvalues.pow(-0.75)) @ eigenvectors.mT
    pinv_preconditioner = (eigenvectors * eigenvalues.reciprocal()) @ eigenvectors.mT
    expected_half = inverse_half_preconditioner @ numerator if shape[0] <= shape[1] \
        else numerator @ inverse_half_preconditioner
    expected_pinv = pinv_preconditioner @ numerator if shape[0] <= shape[1] \
        else numerator @ pinv_preconditioner
    actual_half = muon_from_gram_inverse_half(numerator, gram, epsilon)
    actual_pinv = muon_from_gram_pinv(numerator, gram, epsilon, steps=8)
    torch.testing.assert_close(actual_half, expected_half, rtol=4e-4, atol=5e-5)
    torch.testing.assert_close(actual_pinv, expected_pinv, rtol=3e-3, atol=3e-4)


def test_hybrid_optimizer_routes_matrix_and_auxiliary_parameters():
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = MuonWithAuxAdam([matrix], [bias], lr=1e-2, adam_lr=1e-3,
                                variant="inverse-half", spectral_eps=1e-3)
    matrix_before, bias_before = matrix.detach().clone(), bias.detach().clone()
    (matrix.square().sum() + bias.square().sum()).backward()
    optimizer.step()

    assert not torch.equal(matrix, matrix_before)
    assert not torch.equal(bias, bias_before)
    assert optimizer.param_groups[0]["use_muon"]
    assert not optimizer.param_groups[1]["use_muon"]


def test_pre_ns_cautious_sphere_bias_uses_scalar_agreement_gate():
    torch.manual_seed(23)
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    optimizer = MuonWithAuxAdam(
        [matrix], [], lr=1e-3, adam_lr=1e-3,
        variant="pre-ns-cautious-sphere", momentum=0.0,
        weight_decay=0.0, pre_ns_sphere_strength=0.03,
    )
    matrix.grad = torch.zeros_like(matrix)
    optimizer.step()
    matrix.data.mul_(1.2)
    matrix.grad = matrix.detach().clone()
    optimizer.step()
    diagnostics = optimizer.last_pre_ns_diagnostics
    assert diagnostics["gate_fraction"] == 1.0
    assert diagnostics["abs_log_radius_error_mean"] == pytest.approx(
        torch.log(torch.tensor(1.2)).item(), rel=1e-5)
    assert diagnostics["bias_to_momentum_mean"] == pytest.approx(
        0.03 * torch.log(torch.tensor(1.2)).item(), rel=1e-5)


def test_inverse_half_spectral_diagnostics_match_exact_reference():
    matrix = torch.nn.Parameter(torch.randn(7, 4))
    optimizer = MuonWithAuxAdam(
        [matrix], [], lr=1e-2, adam_lr=1e-3, variant="inverse-half",
        momentum=0.0, weight_decay=0.0, spectral_eps=1e-3,
        rms_scale=1.0, spectral_diagnostics_every=1)
    matrix.grad = torch.randn_like(matrix)
    optimizer.step()
    diagnostics = optimizer.last_spectral_diagnostics
    assert diagnostics["step"] == 1
    assert diagnostics["matrices"] == 1
    assert diagnostics["relative_error_max"] < 5e-4
    assert diagnostics["scale_amplification_max"] > 0
    assert diagnostics["nonfinite"] == 0


def test_attention_operator_muon_lift_matches_rank_tangent_projection():
    torch.manual_seed(20)
    input_size, heads, head_size = 9, 2, 3
    query = torch.randn(heads * head_size, input_size)
    key = torch.randn_like(query)
    query_gradient = torch.empty_like(query)
    key_gradient = torch.empty_like(key)
    ambient_gradients = []
    for head in range(heads):
        rows = slice(head * head_size, (head + 1) * head_size)
        A, B = query[rows].mT, key[rows].mT
        A_plus, B_plus = torch.linalg.pinv(A), torch.linalg.pinv(B)
        PA, PB = A @ A_plus, B @ B_plus
        raw_H = torch.randn(input_size, input_size)
        H = PA @ raw_H + raw_H @ PB - PA @ raw_H @ PB
        ambient_gradients.append(H)
        query_gradient[rows] = (H @ B).mT
        key_gradient[rows] = (H.mT @ A).mT

    query_update, key_update = attention_operator_muon_updates(
        query, key, query_gradient, key_gradient, heads, eps=1e-7)
    for head, H in enumerate(ambient_gradients):
        rows = slice(head * head_size, (head + 1) * head_size)
        A, B = query[rows].mT, key[rows].mT
        A_plus, B_plus = torch.linalg.pinv(A), torch.linalg.pinv(B)
        PA, PB = A @ A_plus, B @ B_plus
        polar = muon_polar(H, 1e-7)
        expected = PA @ polar + polar @ PB - PA @ polar @ PB
        actual = query_update[rows].mT @ B.mT + A @ key_update[rows]
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-3)


def test_attention_operator_optimizer_updates_pair_and_unpaired_matrix():
    torch.manual_seed(21)
    query = torch.nn.Parameter(torch.randn(6, 9))
    key = torch.nn.Parameter(torch.randn(6, 9))
    other = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = MuonWithAuxAdam(
        [query, key, other], [], lr=1e-2, adam_lr=1e-3,
        variant="attention-operator", momentum=0.0, weight_decay=0.0,
        spectral_eps=1e-4, rms_scale=0.2,
        attention_pairs=[(query, key, 2)])
    before = [parameter.detach().clone() for parameter in (query, key, other)]
    query.grad = torch.randn_like(query)
    key.grad = torch.randn_like(key)
    other.grad = torch.randn_like(other)
    optimizer.step()
    for parameter, old in zip((query, key, other), before):
        assert not torch.equal(parameter, old)
    pair_rms = ((query - before[0]).square().sum()
                + (key - before[1]).square().sum()).sqrt() / (
                    query.numel() + key.numel()) ** 0.5
    torch.testing.assert_close(pair_rms, torch.tensor(2e-3), rtol=2e-5, atol=1e-7)


def test_projected_inverse_half_shrinks_only_orthogonal_residual():
    torch.manual_seed(3)
    gradient = torch.randn(7, 4)
    epsilon, beta = 1e-3, 0.25
    polar = muon_polar(gradient, epsilon)
    inverse_half = muon_inverse_half(gradient, epsilon)
    scale = (inverse_half * polar).sum() / polar.square().sum()
    parallel = scale * polar
    expected = parallel + beta * (inverse_half - parallel)
    actual = muon_projected_inverse_half(gradient, epsilon, beta)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    residual = actual - parallel
    torch.testing.assert_close((residual * polar).sum(), torch.tensor(0.0),
                               atol=2e-5, rtol=0.0)
    torch.testing.assert_close(residual.norm(),
                               beta * (inverse_half - parallel).norm(),
                               rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5)])
def test_gated_inverse_half_matches_soft_polar_composition(shape):
    torch.manual_seed(4)
    gradient = torch.randn(shape)
    epsilon = 1e-3
    polar = muon_polar(gradient, epsilon)
    inverse_half = muon_inverse_half(gradient, epsilon)
    expected = (polar @ polar.mT) @ inverse_half if shape[0] <= shape[1] \
        else polar @ (polar.mT @ inverse_half)
    actual = muon_gated_inverse_half(gradient, epsilon)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(5, 8), (8, 5)])
def test_correlation_gated_inverse_half_preserves_requested_order(shape):
    torch.manual_seed(5)
    gradient = torch.randn(shape)
    epsilon = 1e-3
    polar = muon_polar(gradient, epsilon)
    inverse_half = muon_inverse_half(gradient, epsilon)
    expected = polar @ (polar.mT @ inverse_half)
    actual = muon_correlation_gated_inverse_half(gradient, epsilon)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("variant", ["polar", "stable", "polished",
                                     "progressive-polished", "equivariant-polished",
                                     "adaptive", "spectral-sketch-polished",
                                     "split-merge-one", "split-merge-two",
                                     "inverse-half-cubed-quarter",
                                     "inverse-half",
                                     "projected-inverse-half",
                                     "gated-inverse-half",
                                     "correlation-gated-inverse-half",
                                     "randomized-svd-inverse-half",
                                     "randomized-svd-pinv", "ns-pinv",
                                     "ema-gram-inverse-half", "ema-gram-pinv",
                                     "soap-muon-eigh"])
def test_rms_matched_update_has_requested_scale(variant):
    torch.manual_seed(1)
    parameter = torch.nn.Parameter(torch.randn(7, 4))
    before = parameter.detach().clone()
    optimizer = MuonWithAuxAdam(
        [parameter], [], lr=1.0, adam_lr=1e-3, variant=variant,
        momentum=0.0, weight_decay=0.0, spectral_eps=1e-3, rms_scale=0.2)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    update_rms = (parameter - before).norm() / parameter.numel() ** 0.5
    torch.testing.assert_close(update_rms, torch.tensor(0.2), rtol=1e-6, atol=1e-7)


def test_soap_muon_exact_eigh_preserves_preconditioned_frobenius_norm():
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = MuonWithAuxAdam(
        [parameter], [], lr=1.0, adam_lr=1e-3,
        variant="soap-muon-eigh", momentum=0.0, weight_decay=0.0,
        spectral_eps=1e-3, rms_scale=0.2, gram_beta2=0.9)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    torch.testing.assert_close(state["soap_row_basis"].mT @ state["soap_row_basis"],
                               torch.eye(6), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state["soap_col_basis"].mT @ state["soap_col_basis"],
                               torch.eye(4), rtol=1e-5, atol=1e-5)
    parameter.grad = torch.randn_like(parameter)
    before = parameter.detach().clone()
    optimizer.step()
    update_rms = (parameter - before).norm() / parameter.numel() ** 0.5
    torch.testing.assert_close(update_rms, torch.tensor(0.2), rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(state["soap_exp_avg_sq"]) > 0


def test_shared_soap_muon_interface_sums_neighbor_grams_once():
    first = torch.nn.Parameter(torch.randn(4, 3))
    second = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = MuonWithAuxAdam(
        [first, second], [], lr=1e-2, adam_lr=1e-3,
        variant="soap-muon-interface-shared", momentum=0.0,
        weight_decay=0.0, spectral_eps=1e-3, rms_scale=0.2,
        gram_beta2=0.9)
    first.grad = torch.randn_like(first)
    second.grad = torch.randn_like(second)
    first_gram = first.grad @ first.grad.mT
    second_gram = second.grad.mT @ second.grad
    eps = torch.finfo(first_gram.dtype).eps
    first_normalized = first_gram / (first_gram.trace() / 4 + eps)
    second_normalized = second_gram / (second_gram.trace() / 4 + eps)
    expected = 0.1 * (0.5 * first_normalized + 0.5 * second_normalized)
    optimizer.step()
    torch.testing.assert_close(
        optimizer.soap_interface_grams[(0, "shared")], expected)
    assert len(optimizer.soap_interface_grams) == 3


@pytest.mark.parametrize("mix_lambda", [0.25, 0.5, 0.75])
def test_shared_interface_lambda_weights_normalized_neighbors(mix_lambda):
    first = torch.nn.Parameter(torch.randn(4, 3))
    second = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = MuonWithAuxAdam(
        [first, second], [], lr=1e-2, adam_lr=1e-3,
        variant="soap-muon-interface-shared", momentum=0.0,
        weight_decay=0.0, spectral_eps=1e-3, gram_beta2=0.0,
        interface_mix_lambda=mix_lambda)
    first.grad = 100 * torch.randn_like(first)
    second.grad = 0.01 * torch.randn_like(second)
    first_gram = first.grad @ first.grad.mT
    second_gram = second.grad.mT @ second.grad
    expected = (
        mix_lambda * first_gram / (first_gram.trace() / 4 + torch.finfo(first_gram.dtype).eps)
        + (1 - mix_lambda) * second_gram / (second_gram.trace() / 4 + torch.finfo(second_gram.dtype).eps)
    )
    optimizer.step()
    torch.testing.assert_close(
        optimizer.soap_interface_grams[(0, "shared")], expected)


@pytest.mark.parametrize("shape", [(4, 9), (9, 4), (6, 6)])
def test_generalized_half_muon_matches_svd(shape):
    torch.manual_seed(22)
    matrix = torch.randn(shape)
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    damping = 1e-3 * (matrix @ matrix.mT if shape[0] <= shape[1]
                     else matrix.mT @ matrix).norm()
    expected = (u * (singular * (singular.square() + damping).pow(-0.25))) @ vh
    actual = generalized_muon(matrix, 0.5, eps=1e-3)
    torch.testing.assert_close(actual, expected, rtol=4e-4, atol=5e-5)


@pytest.mark.parametrize("power", [0.0, 0.5])
def test_joint_muon_pair_is_split_from_one_shared_transform(power):
    torch.manual_seed(23)
    first = torch.randn(4, 7)
    second = torch.randn(6, 4)
    first_update, second_update = joint_muon_pair_updates(
        first, second, power=power, mix_lambda=0.25,
        normalize_blocks=True, eps=1e-3)
    joined = torch.cat((0.25 ** 0.5 * first / first.norm(),
                        0.75 ** 0.5 * second.mT / second.norm()), dim=1)
    expected = generalized_muon(joined, power, eps=1e-3)
    torch.testing.assert_close(first_update, expected[:, :7])
    torch.testing.assert_close(second_update, expected[:, 7:].mT)


def test_joint_half_muon_diagnostics_detect_accurate_finite_ns_result():
    diagnostics = {}
    joint_muon_pair_updates(
        torch.randn(4, 7), torch.randn(6, 4), power=0.5,
        mix_lambda=0.5, normalize_blocks=True, eps=1e-3,
        diagnostics=diagnostics)
    assert diagnostics["ns_relative_error"] < 5e-4
    assert diagnostics["n_nonfinite"] == 0
    assert 0.0 < diagnostics["first_energy_fraction"] < 1.0


@pytest.mark.parametrize("variant", ["joint-muon-p0", "joint-muon-p05"])
def test_joint_muon_optimizer_uses_only_momentum_state_and_matches_pair_rms(variant):
    first = torch.nn.Parameter(torch.randn(4, 3))
    second = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = MuonWithAuxAdam(
        [first, second], [], lr=1.0, adam_lr=1e-3, variant=variant,
        momentum=0.0, weight_decay=0.0, spectral_eps=1e-3,
        rms_scale=0.2, joint_mix_lambda=0.5)
    before = {id(parameter): parameter.detach().clone()
              for parameter in (first, second)}
    first.grad = torch.randn_like(first)
    second.grad = torch.randn_like(second)
    optimizer.step()
    combined_update_norm = sum(
        (parameter - before[id(parameter)]).square().sum()
        for parameter in (first, second)).sqrt()
    combined_update_rms = combined_update_norm / (
        first.numel() + second.numel()) ** 0.5
    torch.testing.assert_close(
        combined_update_rms, torch.tensor(0.2), rtol=2e-5, atol=1e-6)
    for parameter in (first, second):
        assert set(optimizer.state[parameter]) <= {"momentum_buffer", "step"}


@pytest.mark.parametrize(
    "variant", ["soap-muon-independent", "soap-muon-interface-shared"])
def test_interface_soap_muon_is_finite_and_rms_matched(variant):
    first = torch.nn.Parameter(torch.randn(4, 3))
    second = torch.nn.Parameter(torch.randn(5, 4))
    optimizer = MuonWithAuxAdam(
        [first, second], [], lr=1.0, adam_lr=1e-3, variant=variant,
        momentum=0.0, weight_decay=0.0, spectral_eps=1e-3,
        rms_scale=0.2, gram_beta2=0.9)
    for parameter in (first, second):
        before = parameter.detach().clone()
        parameter.grad = torch.randn_like(parameter)
        parameter._before = before
    optimizer.step()
    for parameter in (first, second):
        update_rms = (parameter - parameter._before).norm() / parameter.numel() ** 0.5
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(
            update_rms, torch.tensor(0.2), rtol=2e-5, atol=1e-6)
