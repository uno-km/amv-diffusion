import time
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from termux_diffusion.core import GPUDutyCycleController, generate
from termux_diffusion.hardware import TermuxExecutionGuard
from termux_diffusion.cli import main


def test_duty_cycle_controller_clamp_and_enable():
    # Test bounds clamping
    ctrl_low = GPUDutyCycleController(target_duty_cycle=30)
    assert ctrl_low.target_duty_cycle == 50
    assert ctrl_low.enabled is True

    ctrl_high = GPUDutyCycleController(target_duty_cycle=120)
    assert ctrl_high.target_duty_cycle == 100
    assert ctrl_high.enabled is False

    ctrl_normal = GPUDutyCycleController(target_duty_cycle=90)
    assert ctrl_normal.target_duty_cycle == 90
    assert ctrl_normal.enabled is True


def test_duty_cycle_controller_on_step():
    ctrl = GPUDutyCycleController(target_duty_cycle=90, enabled=True)
    # Step 1
    gap1 = ctrl.on_step(1, proc_pid=None)
    assert gap1 >= 0.015
    assert ctrl.accumulated_sleep >= 0.015

    # Step 1 repeated (same step should return 0)
    gap1_repeat = ctrl.on_step(1, proc_pid=None)
    assert gap1_repeat == 0.0

    # Step 2
    gap2 = ctrl.on_step(2, proc_pid=None)
    assert gap2 >= 0.015
    assert ctrl.accumulated_sleep >= gap1 + gap2


def test_duty_cycle_controller_disabled_when_100():
    ctrl = GPUDutyCycleController(target_duty_cycle=100, enabled=False)
    gap = ctrl.on_step(1, proc_pid=None)
    assert gap == 0.0
    assert ctrl.accumulated_sleep == 0.0


def test_termux_execution_guard_safe_mode():
    guard = TermuxExecutionGuard(enable_wake_lock=True, boost_priority=True, safe_mode=True)
    assert guard.safe_mode is True
    assert guard.boost_priority is False  # Priority boost disabled when safe_mode active


def test_generate_safe_mode_defaults():
    with patch("termux_diffusion.core.subprocess.Popen") as mock_popen, \
         patch("termux_diffusion.core.locate_sd_cli") as mock_locate, \
         patch("termux_diffusion.core.resolve_model_path", return_value=Path("/models/realistic.gguf")), \
         patch("termux_diffusion.core.check_memory_safety", return_value=(True, "OK")), \
         patch("termux_diffusion.core.export_to_android_gallery", return_value=None):

        mock_locate.return_value = "/bin/sd-cli"
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.stdout = None
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Call with keep_apps_alive=True
        with patch.object(Path, "is_file", return_value=True):
            res = generate(
                prompt="A safe mode test",
                keep_apps_alive=True,
                device="cpu",
                output="safe_test.png"
            )
            # Check mock_popen command arguments had --vae-tiling injected
            call_args = mock_popen.call_args[0][0]
            assert "--vae-tiling" in call_args


def test_cli_safe_mode_args():
    with patch("termux_diffusion.cli.generate") as mock_gen:
        mock_gen.return_value = MagicMock(path=MagicMock(exists=lambda: True))

        # Test --safe-mode flag
        main(["generate", "test prompt", "--safe-mode", "--gpu-duty-cycle", "85"])
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["safe_mode"] is True
        assert call_kwargs["gpu_duty_cycle"] == 85

        # Test --keep-apps-alive alias
        main(["generate", "test prompt", "--keep-apps-alive"])
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["keep_apps_alive"] is True
