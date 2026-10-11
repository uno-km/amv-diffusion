# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
GPU requests must stop with the shared AMU_ERR_RUNTIME_REQUIRED explanation when
ameva-runtime (AMUDA) is missing or too old. The CPU is never a silent fallback.
The runtime is simulated; no GPU and no real ameva-runtime are needed.
"""
import sys
from unittest.mock import patch

import pytest

from termux_diffusion import cli, hardware
from termux_diffusion.exceptions import ExitCode, PlatformNotSupportedError, RuntimeRequiredError


def _runtime(version):
    return patch("ameva_device.runtime_guard.installed_runtime_version", return_value=version)


@pytest.mark.parametrize("device", ["gpu", "vulkan", "opencl"])
def test_explicit_gpu_without_runtime_stops(device):
    with _runtime(None):
        with pytest.raises(RuntimeRequiredError) as info:
            hardware.resolve_device_backend(device)
    text = str(info.value)
    assert isinstance(info.value, PlatformNotSupportedError)
    assert "[AMU_ERR_RUNTIME_REQUIRED] termux-diffusion:" in text
    assert "not installed" in text
    assert "no CPU fallback was used" in text
    assert "AMUDA" in text
    assert 'pip install "ameva-runtime>=2.0.0"' in text
    assert "npm install -g @ameva/runtime" in text
    assert "https://github.com/uno-km/ameva-runtime-releases" in text
    assert "--device cpu" in text


def test_old_runtime_is_rejected():
    with _runtime("1.9.0"):
        with pytest.raises(RuntimeRequiredError) as info:
            hardware.resolve_device_backend("vulkan")
    assert "1.9.0 is installed, but >= 2.0.0 is required" in str(info.value)


def test_cpu_never_touches_the_guard():
    with patch.object(hardware, "_require_gpu_runtime", side_effect=AssertionError("guard called")):
        assert hardware.resolve_device_backend("cpu") == ("cpu", 0)


def test_auto_without_runtime_uses_cpu_and_says_so(capsys):
    with patch.object(hardware, "_resolve_ameva_runtime", return_value=None):
        assert hardware.resolve_device_backend("auto") == ("cpu", 0)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GPU not used" in captured.err
    assert "pip install ameva-runtime" in captured.err


def test_missing_sdk_is_an_explicit_error_not_a_fallback():
    with patch.dict(sys.modules, {"ameva_device": None}):
        with pytest.raises(RuntimeRequiredError) as info:
            hardware.resolve_device_backend("gpu")
    assert "ameva-device-sdk" in str(info.value)
    assert "no CPU fallback was used" in str(info.value)


@pytest.mark.parametrize("flag", [["--gpu"], ["-d", "vulkan"], ["-d", "opencl"]])
def test_cli_exits_69_with_explanation_on_stderr(flag, capsys):
    with _runtime(None):
        code = cli.main(["generate", "a cat"] + flag)
    captured = capsys.readouterr()
    assert code == ExitCode.RUNTIME_REQUIRED == 69
    assert captured.out == ""
    assert "AMU_ERR_RUNTIME_REQUIRED" in captured.err


def test_package_imports_without_numpy_and_pillow():
    import subprocess

    code = (
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "sys.modules['PIL'] = None\n"
        "import termux_diffusion\n"
        "try:\n"
        "    termux_diffusion.refine_face_regions\n"
        "except ImportError:\n"
        "    sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
