"""Temperature scheduling for contrastive learning losses.

Extracted from HDIMTrainer to isolate temperature logic from training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def effective_temperature(
    log_temp: nn.Parameter | None,
    temp_schedule: str,
    current_epoch: int,
    infonce_temperature: float,
    tau_max: float,
    tau_min: float,
    temp_schedule_T_0: int,
) -> float:
    """Compute effective temperature (learnable, scheduled, or fixed).

    Args:
        log_temp: Learnable log-temperature parameter (None if fixed).
        temp_schedule: "none", "warm_restart", etc.
        current_epoch: Current training epoch.
        infonce_temperature: Base/fixed temperature.
        tau_max: Max temperature for warm_restart.
        tau_min: Min temperature for warm_restart.
        temp_schedule_T_0: Cycle length for warm_restart.

    Returns:
        Effective temperature as float.
    """
    if log_temp is not None:
        return float(log_temp.exp().clamp(0.01, 0.5).item())
    if temp_schedule == "warm_restart":
        # Linear decay within current LR restart cycle
        T_0 = temp_schedule_T_0
        epoch_in_cycle = current_epoch % T_0
        fraction = epoch_in_cycle / max(T_0, 1)
        return tau_max - (tau_max - tau_min) * fraction
    return infonce_temperature


def cluster_scaled_temperature(
    embeddings: torch.Tensor,
    base_temp: float,
) -> float:
    """SC-InfoNCE: scale temperature based on cluster tightness (Cheng et al., Nov 2025).

    Tighter clusters -> lower temperature (sharper distribution).
    Looser clusters -> higher temperature (softer distribution).
    """
    if embeddings.numel() == 0:
        return base_temp
    cluster_var = embeddings.var(dim=0, unbiased=False).mean()
    scale = torch.exp(-cluster_var)
    scaled = base_temp * scale.clamp(0.5, 2.0)
    return float(scaled.item())
