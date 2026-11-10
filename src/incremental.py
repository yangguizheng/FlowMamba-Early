"""Incremental inference for FlowMamba-Early: one Mamba-2 state per flow, updated packet by packet.

model_forward_incremental_chunk() feeds one new 320-byte packet record through the
backbone, carrying the convolution and SSM states of every layer over from the previous
packets, and returns the hidden state at the last byte of that packet. An exit read from
this state matches the exit of a full forward pass over the same prefix.

Run this file to compare incremental inference with the full forward pass on a randomly
initialised model in fp32 and bf16:

    python src/incremental.py
"""
import torch
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from run_flowmamba_early import FlowMambaEarly

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mamba2_layer_chunk_step(layer: Mamba2, u, conv_state, ssm_state):
    """Process one chunk `u` (B, L, D) through a single Mamba2 layer, carrying
    (conv_state, ssm_state) across chunk boundaries. Returns (out, new_conv_state, new_ssm_state).
    conv_state/ssm_state may be None on the first chunk."""
    zxbcdt = layer.in_proj(u)
    d_mlp = (zxbcdt.shape[-1] - 2 * layer.d_ssm - 2 * layer.ngroups * layer.d_state - layer.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, layer.d_ssm, layer.d_ssm + 2 * layer.ngroups * layer.d_state, layer.nheads], dim=-1
    )
    xBC_t = xBC.transpose(1, 2)
    xBC_t, conv_state_out = causal_conv1d_fn(
        xBC_t, rearrange(layer.conv1d.weight, "d 1 w -> d w"), bias=layer.conv1d.bias,
        activation=layer.activation, initial_states=conv_state, return_final_states=True,
    )
    xBC = xBC_t.transpose(1, 2)
    x, B, C = torch.split(xBC, [layer.d_ssm, layer.ngroups * layer.d_state, layer.ngroups * layer.d_state], dim=-1)
    A = -torch.exp(layer.A_log.float())
    y, ssm_state_out = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=layer.headdim), dt, A,
        rearrange(B, "b l (g n) -> b l g n", g=layer.ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=layer.ngroups),
        chunk_size=layer.chunk_size,
        D=rearrange(layer.D, "(h p) -> h p", p=layer.headdim) if layer.D_has_hdim else layer.D,
        dt_bias=layer.dt_bias, dt_softplus=True,
        initial_states=ssm_state, return_final_states=True,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    if layer.rmsnorm:
        y = layer.norm(y, z)
    out = layer.out_proj(y)
    return out, conv_state_out, ssm_state_out


class IncrementalState:
    """Convolution and SSM states of every layer for a batch of flows; None before the first packet."""

    def __init__(self, n_layers):
        self.conv = [None] * n_layers
        self.ssm = [None] * n_layers


def model_forward_incremental_chunk(model: FlowMambaEarly, x_chunk, state: IncrementalState):
    """Process one new 320-byte chunk (B, 320) given the prior state, update the state in place,
    and return the hidden state at the last position of this chunk (after norm_f)."""
    h = model.emb(x_chunk)
    for i, (layer, norm) in enumerate(zip(model.layers, model.norms)):
        normed = norm(h)
        out, state.conv[i], state.ssm[i] = mamba2_layer_chunk_step(layer, normed, state.conv[i], state.ssm[i])
        h = h + out
    h = model.norm_f(h)
    return h[:, -1]  # last position of this chunk


def verify_correctness(model, batch=2, K=5, B_bytes=320, dtype=torch.float32):
    """Largest and mean absolute difference between the hidden states of the full forward pass
    and of incremental inference, at the last byte of each of K packets of random bytes."""
    L = K * B_bytes
    x = torch.randint(0, 256, (batch, L), device=DEVICE)
    model = model.to(dtype)
    with torch.no_grad():
        h_full = model.emb(x)
        for layer, norm in zip(model.layers, model.norms):
            h_full = h_full + layer(norm(h_full))
        h_full = model.norm_f(h_full)

        state = IncrementalState(len(model.layers))
        last_positions = []
        for k in range(K):
            chunk = x[:, k * B_bytes:(k + 1) * B_bytes]
            last_h = model_forward_incremental_chunk(model, chunk, state)
            last_positions.append(last_h)
        h_chunked_last = torch.stack(last_positions, dim=1)  # (B, K, D) = h at each packet's last byte

        full_at_boundaries = torch.stack([h_full[:, (k + 1) * B_bytes - 1] for k in range(K)], dim=1)
        diff = (full_at_boundaries.float() - h_chunked_last.float()).abs()
        return diff.max().item(), diff.mean().item()


def main():
    torch.manual_seed(0)
    for dtype, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        model = FlowMambaEarly(num_classes=20, device=DEVICE).to(DEVICE).to(dtype).eval()
        max_diff, mean_diff = verify_correctness(model, batch=4, K=5, dtype=dtype)
        print(f"[{name}] incremental vs full forward at packet boundaries: "
              f"max_abs_diff={max_diff:.6g} mean_abs_diff={mean_diff:.6g}", flush=True)


if __name__ == "__main__":
    main()
