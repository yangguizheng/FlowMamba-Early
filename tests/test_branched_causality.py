"""Causality of the exit-specific branches of run_branched_exits.py. Requires a CUDA GPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_branched_exits import ARR_LENGTH, EXIT_POSITIONS, NUM_PACKET_BYTE, FlowMambaEarlyV3  # noqa: E402


def test_branches_ignore_later_packets():
    torch.manual_seed(0)
    for n_shared, n_branch in [(2, 2), (1, 3)]:
        model = FlowMambaEarlyV3(num_classes=10, n_shared=n_shared, n_branch=n_branch, device="cuda").cuda().eval()
        x = torch.randint(0, 256, (4, ARR_LENGTH), device="cuda")
        with torch.no_grad():
            ref = model(x)
            for i, pos in enumerate(EXIT_POSITIONS):
                y = x.clone()
                y[:, pos * NUM_PACKET_BYTE:] = torch.randint(0, 256, y[:, pos * NUM_PACKET_BYTE:].shape, device="cuda")
                assert (model(y)[i] - ref[i]).abs().max().item() < 1e-5, (n_shared, n_branch, pos)
