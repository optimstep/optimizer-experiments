import math

import torch
import torch.nn as nn

from train_gradcache_qwen3 import (
    apply_matrix_step_control,
    capture_matrix_step_inputs,
    initialize_matrix_step_control,
)


def make_control(mode, ratio, attraction=0.01):
    model = nn.Sequential(nn.Linear(5, 3, bias=True), nn.LayerNorm(3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    control = initialize_matrix_step_control(
        model, optimizer, mode, ratio, attraction)
    return model, optimizer, control


def test_trust_ratio_controls_linear_weight_but_not_bias():
    model, optimizer, control = make_control("trust-ratio", 0.1)
    weight_before = model[0].weight.detach().clone()
    bias_before = model[0].bias.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    apply_matrix_step_control(control, snapshots)
    displacement = model[0].weight.detach() - weight_before
    assert torch.allclose(displacement.norm(), 0.1 * weight_before.norm(), rtol=1e-5)
    assert not torch.allclose(model[0].bias.detach(), bias_before)


def test_fro_sphere_preserves_radius_and_chord_ratio():
    model, optimizer, control = make_control("fro-sphere", 0.1)
    before = model[0].weight.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    metrics = apply_matrix_step_control(control, snapshots, True)
    after = model[0].weight.detach()
    assert torch.allclose(after.norm(), before.norm(), rtol=2e-5, atol=1e-6)
    assert torch.allclose((after - before).norm(), 0.1 * before.norm(),
                          rtol=2e-5, atol=1e-6)
    assert metrics["nonfinite"] == 0
    assert math.isfinite(metrics["raw_radial_fraction"])


def test_learning_rate_schedule_scales_target_ratio():
    model, optimizer, control = make_control("trust-ratio", 0.1)
    optimizer.param_groups[0]["initial_lr"] = 0.2
    optimizer.param_groups[0]["lr"] = 0.05
    before = model[0].weight.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    apply_matrix_step_control(control, snapshots)
    assert torch.allclose((model[0].weight.detach() - before).norm(),
                          0.025 * before.norm(), rtol=1e-5)


def test_sphere_attraction_closes_a_fraction_of_radial_error():
    model, optimizer, control = make_control("sphere-attraction", 0.1, 0.2)
    before = model[0].weight.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    model[0].weight.data.mul_(2.0)
    apply_matrix_step_control(control, snapshots)
    raw = model[0].weight.detach() - before
    trial = before + raw * (0.1 * before.norm() / raw.norm())
    expected_norm = 0.8 * trial.norm() + 0.2 * before.norm()
    assert torch.allclose(model[0].weight.detach().norm(), expected_norm,
                          rtol=1e-5, atol=1e-6)


def test_cautious_sphere_attraction_only_keeps_agreeing_coordinates():
    model, optimizer, control = make_control(
        "cautious-sphere-attraction", 0.1, 0.2)
    before = model[0].weight.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    raw = torch.ones_like(before)
    model[0].weight.data.copy_(before + raw)
    metrics = apply_matrix_step_control(control, snapshots, True)
    controlled = raw * (0.1 * before.norm() / raw.norm())
    trial = before + controlled
    restoring = 0.2 * (before.norm() / trial.norm() - 1.0) * trial
    expected = trial + restoring * (restoring * controlled >= 0)
    assert torch.allclose(model[0].weight.detach(), expected, rtol=1e-5, atol=1e-6)
    assert 0.0 <= metrics["cautious_mask_fraction"] <= 1.0


def test_hyperball_normalizes_full_update_then_restores_weight_norm():
    model, optimizer, control = make_control("hyperball", 0.1)
    before = model[0].weight.detach().clone()
    snapshots = capture_matrix_step_inputs(control)
    raw = torch.randn_like(before)
    model[0].weight.data.copy_(before + raw)
    apply_matrix_step_control(control, snapshots)
    controlled = raw * (0.1 * before.norm() / raw.norm())
    trial = before + controlled
    expected = trial * (before.norm() / trial.norm())
    assert torch.allclose(model[0].weight.detach(), expected,
                          rtol=1e-5, atol=1e-6)
    assert torch.allclose(model[0].weight.detach().norm(), before.norm(),
                          rtol=1e-5, atol=1e-6)

