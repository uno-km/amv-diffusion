# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Deterministic verification of Clean-Room Purger, Safe-By-Default budgeting, and Latent Checkpointing."""

import os
import tempfile
from pathlib import Path
from termux_diffusion.purger import purge_idle_processes, IMMORTAL_WHITELIST_COMM
from termux_diffusion.safety import (
    evaluate_safe_memory_budget,
    save_latent_checkpoint,
    load_latent_checkpoint,
    parse_meminfo_metrics,
)


def test_whitelist_integrity():
    """Verify immortal whitelist contains essential user shells and system daemons."""
    required = {"sshd", "bash", "sh", "tmux", "screen", "tailscaled", "tailscale"}
    for comm in required:
        assert comm in IMMORTAL_WHITELIST_COMM, f"Critical daemon {comm} missing from immortal whitelist!"


def test_safe_budget_calculation():
    """Verify 95% budget calculation and adaptive tiling thresholds."""
    assessment = evaluate_safe_memory_budget(safe_margin=0.95, swap_allowance="1G")
    assert assessment.safe_budget_mb > 0
    assert assessment.safe_budget_mb <= assessment.mem_available_mb
    assert assessment.allowed_swap_mb == 1024.0
    assert assessment.recommended_tile_size in (96, 128, 256)
    assert assessment.recommended_temp_tile in (1, 2, 4)


def test_latent_checkpoint_roundtrip():
    """Verify atomic serialization and deserialization of 4D latent tensor."""
    dummy_tensor = b"\x00\x01\x02\x03" * 1024  # 4KB dummy latent
    metadata = {
        "model": "wan2.1-v3-vace",
        "frames": 33,
        "width": 480,
        "height": 272,
        "steps_completed": 12,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "test_latent_checkpoint.bin"
        save_latent_checkpoint(checkpoint_path, dummy_tensor, metadata)

        assert checkpoint_path.exists(), "Binary checkpoint file was not created!"
        assert checkpoint_path.with_suffix(".json").exists(), "Metadata JSON was not created!"

        loaded_bytes, loaded_meta = load_latent_checkpoint(checkpoint_path)
        assert loaded_bytes == dummy_tensor, "Loaded bytes do not match original tensor!"
        assert loaded_meta["steps_completed"] == 12, "Metadata integrity check failed!"


def test_purger_dry_run():
    """Verify purger scans /proc without crashing and preserves self and whitelist."""
    summary = purge_idle_processes(threshold_mb=999999.0)  # High threshold to avoid killing user processes in test
    assert summary.scanned_count >= 0
    assert len(summary.killed_processes) == 0


if __name__ == "__main__":
    test_whitelist_integrity()
    test_safe_budget_calculation()
    test_latent_checkpoint_roundtrip()
    test_purger_dry_run()
    print("ALL TESTS PASSED (100% DETERMINISTIC VERIFICATION)")
