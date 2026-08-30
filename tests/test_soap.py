from unittest.mock import patch

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from soap import SOAPWithAuxAdam


def test_soap_skips_initial_update_and_uses_eigh_for_every_basis_refresh():
    matrix = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
    optimizer = SOAPWithAuxAdam(
        [matrix], [], lr=1e-2, adam_lr=1e-2,
        betas=(0.95, 0.95), weight_decay=0.0)
    initial = matrix.detach().clone()

    with patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
        matrix.grad = torch.tensor([[0.4, -0.2], [0.1, 0.3]])
        optimizer.step()
        torch.testing.assert_close(matrix, initial)

        state = optimizer.state[matrix]
        torch.testing.assert_close(
            state["left_basis"].mT @ state["left_basis"], torch.eye(2),
            atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            state["right_basis"].mT @ state["right_basis"], torch.eye(2),
            atol=1e-5, rtol=1e-5)

        matrix.grad = torch.tensor([[0.2, 0.5], [-0.4, 0.1]])
        optimizer.step()

    assert eigh.call_count == 4
    assert not torch.equal(matrix, initial)
    assert torch.isfinite(matrix).all()
    assert optimizer.state[matrix]["step"] == 1


def test_auxiliary_parameters_match_adamw():
    actual = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    expected = torch.nn.Parameter(actual.detach().clone())
    optimizer = SOAPWithAuxAdam(
        [], [actual], lr=3e-4, adam_lr=3e-4,
        adam_betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    reference = torch.optim.AdamW(
        [expected], lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01)

    for gradient in (
        torch.tensor([0.3, -0.1, 0.2]),
        torch.tensor([-0.2, 0.4, 0.1]),
    ):
        actual.grad = gradient.clone()
        expected.grad = gradient.clone()
        optimizer.step()
        reference.step()

    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)

