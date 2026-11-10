"""The masked multi-exit loss: exit N is trained only on flows with at least N packets. Requires a CUDA GPU."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_flowmamba_early import ARR_LENGTH, EXIT_POSITIONS, FlowMambaEarly, masked_train_loss  # noqa: E402


def setup():
    torch.manual_seed(0)
    model = FlowMambaEarly(num_classes=5, device="cuda").cuda()
    x = torch.randint(0, 256, (4, ARR_LENGTH), device="cuda")
    y = torch.tensor([0, 1, 2, 3], device="cuda")
    npkt = torch.tensor([1, 1, 5, 5], device="cuda")
    return model, x, y, npkt


def test_loss_uses_eligible_flows_only():
    model, x, y, npkt = setup()
    outs = model(x)
    # the first two flows have one packet and reach exit 1 only
    expected = (F.cross_entropy(outs[0], y) + sum(F.cross_entropy(outs[j][2:], y[2:]) for j in range(1, 4))) / 4
    assert torch.allclose(masked_train_loss(outs, y, npkt, EXIT_POSITIONS), expected)


def test_exit_loss_reaches_only_its_head():
    model, x, y, npkt = setup()
    outs = model(x)
    j5 = EXIT_POSITIONS.index(5)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(outs[j5][npkt >= 5], y[npkt >= 5]).backward()
    for pos, head in zip(EXIT_POSITIONS, model.heads):
        g = head.weight.grad
        assert (g is not None and g.norm().item() > 0) == (pos == 5), pos
