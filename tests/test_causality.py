"""Causality of FlowMamba-Early: exit N reads only the first N packets. Requires a CUDA GPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_flowmamba_early import ARR_LENGTH, EXIT_POSITIONS, NUM_PACKET_BYTE, FlowMambaEarly  # noqa: E402


def model_and_input():
    torch.manual_seed(0)
    model = FlowMambaEarly(num_classes=5, device="cuda").cuda().eval()
    return model, torch.randint(0, 256, (2, ARR_LENGTH), device="cuda")


def test_exit_ignores_later_packets():
    model, x = model_and_input()
    with torch.no_grad():
        ref = model(x)
        for i, pos in enumerate(EXIT_POSITIONS):
            y = x.clone()
            y[:, pos * NUM_PACKET_BYTE:] = torch.randint(0, 256, y[:, pos * NUM_PACKET_BYTE:].shape, device="cuda")
            assert torch.allclose(model(y)[i], ref[i], atol=1e-5), pos


def test_exit_reads_its_own_packets():
    model, x = model_and_input()
    y = x.clone()
    y[:, :NUM_PACKET_BYTE] = torch.randint(0, 256, (2, NUM_PACKET_BYTE), device="cuda")
    with torch.no_grad():
        assert not torch.allclose(model(x)[0], model(y)[0], atol=1e-5)
