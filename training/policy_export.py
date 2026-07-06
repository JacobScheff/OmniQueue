"""Shared TorchScript export for C++ LibTorch policy inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model import ParkRouterModel, obs_flat_to_tensors
from training.checkpoint import load_checkpoint
from training.features import FLAT_OBS_DIM, NUM_ACTIONS


class PolicyTorchScript(nn.Module):
    def __init__(self, model: ParkRouterModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, obs_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        guest, ride, env = obs_flat_to_tensors(obs_flat)
        logits, value = self.model(guest, ride, env)
        return logits[:, 0, :], value.squeeze(-1)


def torchscript_path_for(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_suffix(".ts.pt")


def export_torchscript(
    checkpoint: str | Path,
    output: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    checkpoint = Path(checkpoint)
    output_path = Path(output) if output is not None else torchscript_path_for(checkpoint)

    model, _step, _meta = load_checkpoint(checkpoint, device)
    model.eval()
    wrapper = PolicyTorchScript(model).eval()

    example = torch.zeros(2, FLAT_OBS_DIM, dtype=torch.float32)
    with torch.no_grad():
        eager_logits, eager_value = wrapper(example)

    try:
        scripted = torch.jit.script(wrapper)
    except Exception:
        scripted = torch.jit.trace(wrapper, example)

    with torch.no_grad():
        script_logits, script_value = scripted(example)
    if not torch.allclose(eager_logits, script_logits, atol=1e-5, rtol=1e-4):
        raise RuntimeError("TorchScript logits mismatch vs eager model.")
    if not torch.allclose(eager_value, script_value, atol=1e-5, rtol=1e-4):
        raise RuntimeError("TorchScript value mismatch vs eager model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    return output_path


def export_model_torchscript(
    model: ParkRouterModel,
    output: str | Path,
    device: str | torch.device = "cpu",
) -> Path:
    output_path = Path(output)
    model = model.to(device)
    model.eval()
    wrapper = PolicyTorchScript(model).eval()
    example = torch.zeros(2, FLAT_OBS_DIM, dtype=torch.float32, device=device)
    try:
        scripted = torch.jit.script(wrapper)
    except Exception:
        scripted = torch.jit.trace(wrapper, example.cpu())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    return output_path
