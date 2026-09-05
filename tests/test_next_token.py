import pytest
import torch

from experiments.next_token.train import (
    ResidualActivationWeightController,
    TinyDecoderLM,
    make_optimizer,
    residual_l1_probe,
    make_lr_multiplier,
)


def make_model():
    return TinyDecoderLM(64, 32, 2, 4, 16, 32).eval()


def controlled_step(model, tokens, targets, strength):
    controller = ResidualActivationWeightController(
        model, targets, ratio=0.008, strength=strength, log_clip=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    model(tokens).square().mean().backward()
    snapshots = controller.capture()
    optimizer.step()
    diagnostics = controller.apply(snapshots)
    controller.close()
    return diagnostics


def test_weight_feedback_changes_trajectory_without_changing_step_budget():
    torch.manual_seed(1)
    control, treatment = make_model(), make_model()
    treatment.load_state_dict(control.state_dict())
    tokens = torch.randint(0, 64, (2, 16))
    measured = residual_l1_probe(control, tokens)
    targets = {name: value * 0.5 for name, value in measured.items()}
    control_diag = controlled_step(control, tokens, targets, strength=0.0)
    treatment_diag = controlled_step(treatment, tokens, targets, strength=1.0)
    assert control_diag["applied_relative_fro"] == pytest.approx(0.008)
    assert treatment_diag["applied_relative_fro"] == pytest.approx(0.008)
    assert control_diag["correction_relative_fro"] == 0.0
    assert treatment_diag["correction_relative_fro"] > 0.0
    assert any(not torch.equal(left, right) for left, right in
               zip(control.parameters(), treatment.parameters()))


def test_weight_feedback_is_finite_and_records_activation_error():
    torch.manual_seed(2)
    model = make_model()
    tokens = torch.randint(0, 64, (2, 16))
    measured = residual_l1_probe(model, tokens)
    targets = {name: value * 1.5 for name, value in measured.items()}
    controller = ResidualActivationWeightController(
        model, targets, ratio=0.008, strength=1.0, log_clip=1.0)
    model(tokens)
    assert all(values["relative_error"] < 0 for values in controller.last.values())
    controller.close()
    diagnostics = controlled_step(model, tokens, targets, strength=1.0)
    assert diagnostics["nonfinite"] == 0
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_controller_strength_can_be_scheduled():
    model = TinyDecoderLM(64, 16, 2, 2, 8, 16)
    tokens = torch.randint(0, 64, (2, 8))
    controller = ResidualActivationWeightController(
        model, residual_l1_probe(model, tokens), 0.008, 12.0, 1.0)
    controller.set_strength(4.0)
    assert controller.strength == 4.0
    controller.close()


def test_hard_controller_preserves_step_budget():
    model = make_model()
    tokens = torch.randint(0, 64, (2, 16))
    targets = {name: value * 0.5
               for name, value in residual_l1_probe(model, tokens).items()}
    controller = ResidualActivationWeightController(
        model, targets, 0.008, 1.0, 1.0, hard=True)
    model(tokens)
    snapshots = controller.capture()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 1e-4)
    diagnostics = controller.apply(snapshots)
    assert diagnostics["applied_relative_fro"] == pytest.approx(0.008)
    assert diagnostics["nonfinite"] == 0
    controller.close()


def test_layer_zero_feedback_controls_residual_source_weights():
    model = make_model()
    targets = residual_l1_probe(model, torch.randint(0, 64, (2, 16)))
    controller = ResidualActivationWeightController(
        model, targets, 0.008, 4.0, 1.0)
    assert controller.parameter_stages[id(model.token.weight)] == "layer_0"
    assert controller.parameter_stages[id(model.position.weight)] == "layer_0"
    controller.close()


def test_residual_source_feedback_has_separate_strength():
    model = make_model()
    targets = residual_l1_probe(model, torch.randint(0, 64, (2, 16)))
    controller = ResidualActivationWeightController(
        model, targets, 0.008, 12.0, 1.0, source_strength=0.1)
    assert controller.parameter_strengths[id(model.token.weight)] == 0.1
    assert controller.parameter_strengths[id(model.position.weight)] == 0.1
    controller.close()


def test_muon_routes_hidden_matrices_and_keeps_embeddings_auxiliary():
    model = make_model()
    config = {"optimizer": {"algorithm": "muon", "lr": 3e-4,
                            "muon_lr": 0.02, "matrix_step_ratio": 0.008}}
    optimizer = make_optimizer(model, config)
    muon_ids = {id(parameter) for group in optimizer.param_groups
                if group["use_muon"] for parameter in group["params"]}
    assert id(model.token.weight) not in muon_ids
    assert id(model.position.weight) not in muon_ids
    assert id(model.blocks.layers[0].linear1.weight) in muon_ids


def test_stable_muon_variant_is_configurable():
    model = make_model()
    config = {"optimizer": {"algorithm": "muon", "muon_variant": "stable",
                            "lr": 3e-4, "matrix_step_ratio": 0.008}}
    optimizer = make_optimizer(model, config)
    assert all(group["variant"] == "stable" for group in optimizer.param_groups)


def test_wsd_schedule_is_shared_by_lr_and_relative_step_target():
    multiplier = make_lr_multiplier({
        "steps": 100, "schedule": "wsd", "warmup_frac": 0.1,
        "decay_frac": 0.2,
    })
    assert multiplier(0) == pytest.approx(0.1)
    assert multiplier(9) == pytest.approx(1.0)
    assert multiplier(50) == pytest.approx(1.0)
    assert multiplier(90) == pytest.approx(0.5)
    assert multiplier(100) == pytest.approx(0.0)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown training schedule"):
        make_lr_multiplier({"steps": 10, "schedule": "mystery"})


def test_controller_scales_rms_over_rms_target_with_schedule():
    model = make_model()
    tokens = torch.randint(0, 64, (2, 16))
    targets = residual_l1_probe(model, tokens)
    controller = ResidualActivationWeightController(
        model, targets, ratio=0.008, strength=0.0, log_clip=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    model(tokens).square().mean().backward()
    snapshots = controller.capture()
    optimizer.step()
    diagnostics = controller.apply(snapshots, ratio_scale=0.25)
    assert diagnostics["applied_relative_fro"] == pytest.approx(
        0.002, rel=1e-5)
    controller.close()

