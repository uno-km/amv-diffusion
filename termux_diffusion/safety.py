# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Deterministic memory budgeting, swap allowance, adaptive tiling, and crash-resilient latent checkpointing.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("termux_diffusion.safety")


@dataclass
class SafeMemoryAssessment:
    mem_total_mb: float
    mem_available_mb: float
    safe_budget_mb: float
    swap_total_mb: float
    swap_free_mb: float
    allowed_swap_mb: float
    recommended_tile_size: int
    recommended_temp_tile: int

    def summary(self) -> str:
        return (
            f"MemAvail={self.mem_available_mb:.1f}MB, "
            f"SafeBudget(95%)={self.safe_budget_mb:.1f}MB, "
            f"AllowedSwap={self.allowed_swap_mb:.1f}MB, "
            f"Tiling={self.recommended_tile_size}px (temp={self.recommended_temp_tile}f)"
        )


def parse_meminfo_metrics() -> Tuple[float, float, float, float]:
    """Parse MemTotal, MemAvailable, SwapTotal, SwapFree deterministically from /proc/meminfo in MB."""
    meminfo = Path("/proc/meminfo")
    total_mb = avail_mb = swap_tot_mb = swap_free_mb = 0.0

    if meminfo.exists():
        try:
            with open(meminfo, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        val_kb = float(parts[1])
                        if key == "MemTotal:":
                            total_mb = val_kb / 1024.0
                        elif key == "MemAvailable:":
                            avail_mb = val_kb / 1024.0
                        elif key == "SwapTotal:":
                            swap_tot_mb = val_kb / 1024.0
                        elif key == "SwapFree:":
                            swap_free_mb = val_kb / 1024.0
        except Exception as e:
            logger.warning("Failed parsing /proc/meminfo: %s", e)

    # Conservative fallback if /proc/meminfo is inaccessible (e.g. non-Linux/testing)
    if avail_mb == 0.0:
        total_mb, avail_mb = 4096.0, 2048.0

    return round(total_mb, 1), round(avail_mb, 1), round(swap_tot_mb, 1), round(swap_free_mb, 1)


def evaluate_safe_memory_budget(
    safe_margin: float = 0.95,
    swap_allowance: str = "auto"
) -> SafeMemoryAssessment:
    """Evaluate 95% safe budget (5% OS reserve) and dynamic VAE tiling bounds."""
    tot, avail, s_tot, s_free = parse_meminfo_metrics()
    safe_budget = round(avail * safe_margin, 1)

    # 스왑 허용치 계산
    allowed_swap = 0.0
    s_lower = str(swap_allowance).strip().lower()
    if s_lower == "auto":
        allowed_swap = round(s_free * 0.5, 1)
    elif s_lower.endswith(("g", "gb")):
        num_str = s_lower.rstrip("gb")
        allowed_swap = round(float(num_str) * 1024.0, 1)
    elif s_lower.endswith(("m", "mb")):
        num_str = s_lower.rstrip("mb")
        allowed_swap = round(float(num_str), 1)
    elif s_lower.isdigit():
        allowed_swap = round(float(s_lower), 1)

    # VAE 타일링 권장치 산출 (메모리 가용량에 따른 자동 Graceful Degradation)
    if safe_budget >= 3072.0:
        tile_size = 256
        temp_tile = 4
    elif safe_budget >= 1843.0:
        tile_size = 128
        temp_tile = 2
    else:
        tile_size = 96
        temp_tile = 1

    return SafeMemoryAssessment(
        mem_total_mb=tot,
        mem_available_mb=avail,
        safe_budget_mb=safe_budget,
        swap_total_mb=s_tot,
        swap_free_mb=s_free,
        allowed_swap_mb=allowed_swap,
        recommended_tile_size=tile_size,
        recommended_temp_tile=temp_tile,
    )


def save_latent_checkpoint(checkpoint_path: Path, latent_bytes: bytes, metadata: Dict[str, Any]) -> None:
    """Atomically persist 4D latent tensor to protect against post-DiT VAE decode crash."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_path.with_suffix(".json")

    temp_bin = checkpoint_path.with_suffix(".tmp.bin")
    with open(temp_bin, "wb") as f:
        f.write(latent_bytes)
    temp_bin.replace(checkpoint_path)

    temp_meta = meta_path.with_suffix(".tmp.json")
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    temp_meta.replace(meta_path)

    logger.info("Latent checkpoint saved atomically: %s (%d bytes)", checkpoint_path, len(latent_bytes))


def load_latent_checkpoint(checkpoint_path: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Load cached 4D latent tensor to bypass DiT inference and execute VAE decode directly."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Latent checkpoint file not found: {checkpoint_path}")

    meta_path = checkpoint_path.with_suffix(".json")
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to parse latent checkpoint metadata: %s", e)

    with open(checkpoint_path, "rb") as f:
        latent_bytes = f.read()

    logger.info("Latent checkpoint loaded: %s (%d bytes)", checkpoint_path, len(latent_bytes))
    return latent_bytes, metadata
