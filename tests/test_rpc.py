"""Unit tests for distributed RPC and multi-node tensor splitting in termux-diffusion."""

import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from termux_diffusion.core import generate, GenerationResult
from termux_diffusion.engine import DiffusionEngine, get_binary_path
from termux_diffusion.cli import main


def test_cli_rpc_and_tensor_split_args(tmp_path):
    """Verify CLI parses --rpc and -ts flags and passes them to generate()."""
    dummy_out = tmp_path / "test.png"
    dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch("termux_diffusion.cli.generate") as mock_generate:
        mock_result = MagicMock(spec=GenerationResult)
        mock_result.path = dummy_out
        mock_generate.return_value = mock_result

        code = main([
            "generate",
            "A cute cat in cyberpunk city",
            "--rpc", "192.168.0.220:50052,192.168.0.253:50052",
            "-ts", "50,50",
            "-m", "speed",
            "-s", "4"
        ])

        assert code == 0
        assert mock_generate.called
        kwargs = mock_generate.call_args[1]
        assert kwargs["rpc"] == "192.168.0.220:50052,192.168.0.253:50052"
        assert kwargs["tensor_split"] == "50,50"


def test_core_generate_builds_rpc_cmd(tmp_path):
    """Verify core.generate injects --rpc and --tensor-split into sd-cli argv."""
    dummy_out = tmp_path / "out.png"
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 64)

    mock_proc = MagicMock()
    mock_proc.stdout = ["Step 1/1 (100%)\n"]
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = ("", "")

    def fake_wait(timeout=None):
        dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return 0
    mock_proc.wait = fake_wait

    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return mock_proc

    with patch("termux_diffusion.hardware.resolve_device_backend", return_value=("cpu", 0)), \
         patch("termux_diffusion.core.check_memory_safety", return_value=(True, "OK")), \
         patch("termux_diffusion.core.locate_sd_cli", return_value=tmp_path / "sd-cli"), \
         patch("termux_diffusion.core.resolve_model_path", return_value=dummy_model), \
         patch("termux_diffusion.core.verify_rpc_cluster_nodes", return_value=None), \
         patch("subprocess.Popen", side_effect=fake_popen):

        res = generate(
            prompt="Cyberpunk robot portrait",
            output=dummy_out,
            rpc="192.168.0.220:50052",
            tensor_split="row",
            export_gallery=False,
            wake_lock=False,
            low_ram_guard=False,
        )

        assert res.rpc == "192.168.0.220:50052"
        assert res.tensor_split == "row"
        assert "--rpc-servers" in captured_cmd
        rpc_idx = captured_cmd.index("--rpc-servers")
        assert captured_cmd[rpc_idx + 1] == "192.168.0.220:50052"
        assert "--split-mode" in captured_cmd
        ts_idx = captured_cmd.index("--split-mode")
        assert captured_cmd[ts_idx + 1] == "row"


def test_diffusion_engine_rpc_propagation():
    """Verify DiffusionEngine stores rpc/tensor_split and forwards to generate()."""
    engine = DiffusionEngine(
        model="realistic",
        backend="auto",
        rpc="192.168.0.253:50052",
        tensor_split="50,50"
    )

    assert engine.rpc == "192.168.0.253:50052"
    assert engine.tensor_split == "50,50"

    with patch("termux_diffusion.core.generate") as mock_gen:
        engine.generate("A majestic mountain")
        assert mock_gen.called
        kwargs = mock_gen.call_args[1]
        assert kwargs["rpc"] == "192.168.0.253:50052"
        assert kwargs["tensor_split"] == "50,50"


def test_engine_binary_path_rpc_lookup(tmp_path):
    """Verify get_binary_path('rpc') searches RPC candidates correctly."""
    with patch("shutil.which", return_value=str(tmp_path / "sd-cli-rpc")), \
         patch("os.access", return_value=True):
        p = get_binary_path("rpc")
        assert p is not None
        assert "sd-cli-rpc" in str(p)


def test_parse_cluster_rpc_spec():
    """Verify cluster endpoint parsing and strict validation."""
    from termux_diffusion.cluster import parse_cluster_rpc_spec
    from termux_diffusion.exceptions import ClusterConfigurationError

    # Valid string
    nodes = parse_cluster_rpc_spec("192.0.2.11:50052, 192.0.2.12:50052")
    assert nodes == ["192.0.2.11:50052", "192.0.2.12:50052"]

    # Valid list
    nodes_list = parse_cluster_rpc_spec(["192.0.2.11:50052", "192.0.2.12:50052"])
    assert nodes_list == ["192.0.2.11:50052", "192.0.2.12:50052"]

    # Invalid: missing port
    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("192.0.2.11")

    # Invalid: bad port
    with pytest.raises(ClusterConfigurationError):
        parse_cluster_rpc_spec("192.0.2.11:99999")


def test_verify_rpc_cluster_nodes_failfast():
    """Verify unreachable cluster node triggers ClusterConnectionError under Zero-Silent-Fallback."""
    from termux_diffusion.cluster import verify_rpc_cluster_nodes
    from termux_diffusion.exceptions import ClusterConnectionError

    # Unreachable local mock port
    with pytest.raises(ClusterConnectionError) as exc_info:
        verify_rpc_cluster_nodes(["127.0.0.1:59999"], timeout=0.1)

    assert "Zero-Silent-Fallback Violation Prevented" in str(exc_info.value)


def test_core_generate_unified_cluster_args(tmp_path):
    """Verify core.generate injects unified cluster parameters into sd-cli argv."""
    dummy_out = tmp_path / "out.png"
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 64)

    mock_proc = MagicMock()
    mock_proc.stdout = ["Step 1/1 (100%)\n"]
    mock_proc.returncode = 0
    mock_proc.communicate.return_value = ("", "")
    mock_proc.wait = lambda timeout=None: 0

    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return mock_proc

    with patch("termux_diffusion.hardware.resolve_device_backend", return_value=("cpu", 0)), \
         patch("termux_diffusion.core.check_memory_safety", return_value=(True, "OK")), \
         patch("termux_diffusion.core.locate_sd_cli", return_value=tmp_path / "sd-cli"), \
         patch("termux_diffusion.core.resolve_model_path", return_value=dummy_model), \
         patch("termux_diffusion.core.verify_rpc_cluster_nodes", return_value=None), \
         patch("subprocess.Popen", side_effect=fake_popen):

        res = generate(
            prompt="Cyberpunk portrait",
            output=dummy_out,
            cluster_rpc_servers=["192.0.2.11:50052", "192.0.2.12:50052"],
            cluster_split_mode="layer",
            cluster_vram_budget="cuda0=6,vulkan0=4",
            export_gallery=False,
            wake_lock=False,
            low_ram_guard=False,
        )

        assert res.cluster_rpc_servers == ["192.0.2.11:50052", "192.0.2.12:50052"]
        assert res.cluster_split_mode == "layer"
        assert res.cluster_vram_budget == "cuda0=6,vulkan0=4"

        assert "--rpc-servers" in captured_cmd
        rpc_idx = captured_cmd.index("--rpc-servers")
        assert captured_cmd[rpc_idx + 1] == "192.0.2.11:50052,192.0.2.12:50052"

        assert "--split-mode" in captured_cmd
        sm_idx = captured_cmd.index("--split-mode")
        assert captured_cmd[sm_idx + 1] == "layer"

        assert "--max-vram" in captured_cmd
        mv_idx = captured_cmd.index("--max-vram")
        assert captured_cmd[mv_idx + 1] == "cuda0=6,vulkan0=4"


def test_cluster_option_forwarder_lossless():
    """Verify ClusterOptionForwarder normalizes options without silent drop and preserves extra_args."""
    from termux_diffusion.cluster import ClusterOptionForwarder

    options = {
        "flow_shift": 3.0,
        "schedule": "simple",
        "steps": 8,
        "cfg_scale": 1.0,
        "diffusion_fa": True,
        "empty_flag": None,
    }
    extra_args = ["--custom-worker-arg", "opt_val", "--dry-run"]

    forwarded = ClusterOptionForwarder.normalize_forward_args(options, extra_args=extra_args)

    assert "--flow-shift" in forwarded
    assert forwarded[forwarded.index("--flow-shift") + 1] == "3.0"
    assert "--schedule" in forwarded
    assert forwarded[forwarded.index("--schedule") + 1] == "simple"
    assert "--steps" in forwarded
    assert forwarded[forwarded.index("--steps") + 1] == "8"
    assert "--cfg-scale" in forwarded
    assert forwarded[forwarded.index("--cfg-scale") + 1] == "1.0"
    assert "--diffusion-fa" in forwarded
    assert "--empty-flag" not in forwarded

    # Check lossless pass-through of extra_args
    assert "--custom-worker-arg" in forwarded
    assert forwarded[forwarded.index("--custom-worker-arg") + 1] == "opt_val"
    assert "--dry-run" in forwarded


def test_sequential_cluster_pipeline_execution():
    """Verify SequentialClusterPipeline executes Stage 1 then Stage 2 with fail-fast enforcement."""
    from termux_diffusion.cluster import SequentialClusterPipeline

    pipeline = SequentialClusterPipeline()
    execution_trace = []

    def mock_stage1():
        execution_trace.append("stage1_text_encoder_run")
        execution_trace.append("stage1_text_encoder_killed")
        return {"cache_hit": True}

    def mock_stage2():
        execution_trace.append("stage2_dit_sampling_run")
        return "final_image.png"

    result = pipeline.execute_stages(
        prompt="A beautiful landscape",
        stage1_fn=mock_stage1,
        stage2_fn=mock_stage2,
    )

    assert result == "final_image.png"
    assert execution_trace == [
        "stage1_text_encoder_run",
        "stage1_text_encoder_killed",
        "stage2_dit_sampling_run",
    ]


def test_core_generate_pass_through_extra_args(tmp_path):
    """Verify generate() forwards extra_args and extra_kwargs into underlying engine command."""
    dummy_out = tmp_path / "out.png"
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 64)

    mock_proc = MagicMock()
    mock_proc.stdout = ["Step 1/1 (100%)\n"]
    mock_proc.returncode = 0
    mock_proc.wait = lambda timeout=None: 0

    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return mock_proc

    with patch("termux_diffusion.hardware.resolve_device_backend", return_value=("cpu", 0)), \
         patch("termux_diffusion.core.check_memory_safety", return_value=(True, "OK")), \
         patch("termux_diffusion.core.locate_sd_cli", return_value=tmp_path / "sd-cli"), \
         patch("termux_diffusion.core.resolve_model_path", return_value=dummy_model), \
         patch("subprocess.Popen", side_effect=fake_popen):

        res = generate(
            prompt="Futuristic city",
            output=dummy_out,
            extra_args=["--high-noise-steps", "4", "--custom-experimental-flag"],
            custom_option_param="test_val",
            export_gallery=False,
            wake_lock=False,
            low_ram_guard=False,
        )

        assert res.extra_args is not None
        assert "--high-noise-steps" in captured_cmd
        hns_idx = captured_cmd.index("--high-noise-steps")
        assert captured_cmd[hns_idx + 1] == "4"
        assert "--custom-experimental-flag" in captured_cmd
        assert "--custom-option-param" in captured_cmd
        cop_idx = captured_cmd.index("--custom-option-param")
        assert captured_cmd[cop_idx + 1] == "test_val"


def test_cli_pass_through_unknown_args(tmp_path):
    """Verify CLI accepts unmodeled engine options via parse_known_args and forwards them to generate()."""
    dummy_out = tmp_path / "test.png"
    dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch("termux_diffusion.cli.generate") as mock_generate:
        mock_result = MagicMock(spec=GenerationResult)
        mock_result.path = dummy_out
        mock_generate.return_value = mock_result

        code = main([
            "generate",
            "A majestic eagle in mountain sunset",
            "--isolate-text-encoder",
            "--custom-subengine-flag", "hello",
            "--unmodeled-opt",
        ])

        assert code == 0
        assert mock_generate.called
        kwargs = mock_generate.call_args[1]
        assert kwargs["isolate_text_encoder"] is True
        assert kwargs["extra_args"] == ["--custom-subengine-flag", "hello", "--unmodeled-opt"]


def test_resolve_auto_tensor_split():
    """Verify resolve_auto_tensor_split resolves 'auto' into valid comma-separated split ratios."""
    from termux_diffusion.cluster import resolve_auto_tensor_split

    # When explicit, keep as-is
    srvs, ts = resolve_auto_tensor_split("192.0.2.10:50052,192.0.2.11:50052", tensor_split="40,30,30")
    assert srvs == ["192.0.2.10:50052", "192.0.2.11:50052"]
    assert ts == "40,30,30"

    # When auto, calculate dynamically
    srvs, ts = resolve_auto_tensor_split("192.0.2.10:50052,192.0.2.11:50052", tensor_split="auto")
    assert srvs == ["192.0.2.10:50052", "192.0.2.11:50052"]
    assert ts is not None
    parts = [int(p) for p in ts.split(",")]
    assert len(parts) == 3
    assert sum(parts) == 100

