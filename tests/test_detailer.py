"""Unit tests for Mobile FaceDetailer module (termux_diffusion/detailer.py)."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock

from termux_diffusion.detailer import (
    calculate_iou,
    non_max_suppression,
    create_feathered_mask,
    detect_faces_on_image,
    refine_face_regions,
)
from termux_diffusion.core import generate, VALID_SCHEDULERS, VALID_SAMPLERS


def test_iou_calculation():
    # Identical boxes -> IoU = 1.0
    b1 = (10, 10, 50, 50)
    assert calculate_iou(b1, b1) == pytest.approx(1.0)

    # Disjoint boxes -> IoU = 0.0
    b2 = (60, 60, 100, 100)
    assert calculate_iou(b1, b2) == pytest.approx(0.0)

    # Half overlap
    b3 = (10, 10, 50, 30)  # area = 40 * 20 = 800
    b4 = (10, 10, 50, 50)  # area = 40 * 40 = 1600
    # inter = 40 * 20 = 800, union = 1600, IoU = 0.5
    assert calculate_iou(b3, b4) == pytest.approx(0.5)


def test_non_max_suppression():
    # Two almost identical boxes
    b1 = (10, 10, 50, 50)
    b2 = (12, 11, 49, 51)
    # Distinct box elsewhere
    b3 = (200, 200, 250, 250)

    merged = non_max_suppression([b1, b2, b3], iou_threshold=0.3)
    assert len(merged) == 2
    assert b3 in merged


def test_create_feathered_mask():
    mask = create_feathered_mask((100, 100), blur_radius=5.0)
    assert mask.size == (100, 100)
    assert mask.mode == "L"

    # Center should be high value (opaque), outer edge should be low (transparent)
    np_mask = np.array(mask)
    assert np_mask[50, 50] > 200
    assert np_mask[0, 0] < 50


def test_detect_faces_mock(tmp_path):
    # Create synthetic test image
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw two dark squares simulating face candidates
    draw.rectangle([30, 30, 80, 80], fill=(50, 50, 50))
    draw.rectangle([120, 120, 170, 170], fill=(50, 50, 50))
    img_path = tmp_path / "test_detect.png"
    img.save(img_path)

    # Mock termux_vision to return detections
    mock_bbox = MagicMock()
    mock_bbox.left = 30
    mock_bbox.top = 30
    mock_bbox.right = 80
    mock_bbox.bottom = 80
    mock_det = MagicMock(bbox=mock_bbox)

    with patch.dict("sys.modules", {"termux_vision": MagicMock(detect_faces=lambda _: [mock_det])}):
        boxes = detect_faces_on_image(img_path, min_face_size=20)
        assert len(boxes) == 1
        assert boxes[0] == (30, 30, 80, 80)


def test_refine_face_regions_pipeline(tmp_path):
    # Create a base 512x512 image
    base_img = Image.new("RGB", (512, 512), color=(200, 180, 160))
    base_path = tmp_path / "base.png"
    out_path = tmp_path / "detailed.png"
    base_img.save(base_path)

    # Mock face detection to return 1 face box
    mock_boxes = [(100, 100, 200, 200)]

    def fake_generator(prompt, init_img, output, **kwargs):
        # Fake img2img generation: create refined crop
        refined = Image.new("RGB", (512, 512), color=(255, 200, 180))
        refined.save(output, "PNG")

    with patch("termux_diffusion.detailer.detect_faces_on_image", return_value=mock_boxes):
        result_path = refine_face_regions(
            image_path=base_path,
            generator_func=fake_generator,
            prompt="korean family",
            output_path=out_path,
            strength=0.35,
            steps=2,
            padding_factor=1.4
        )
        assert result_path == out_path
        assert out_path.is_file()

        final_img = Image.open(out_path)
        assert final_img.size == (512, 512)


def test_flux_scheduler_in_valid_schedulers():
    assert "flux" in VALID_SCHEDULERS
    assert "flux2" in VALID_SCHEDULERS
    assert "euler" in VALID_SAMPLERS
    assert "euler_a" in VALID_SAMPLERS


def test_z_image_turbo_auto_scheduler_flux(tmp_path):
    """Verify that z-image-turbo automatically assigns --scheduler flux and --guidance 3.5."""
    dummy_model = tmp_path / "z_image_turbo-Q2_K.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 32)
    dummy_out = tmp_path / "out.png"

    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        mock_proc = MagicMock()
        mock_proc.stdout = ["Step 1/1 (100%)\n"]
        mock_proc.returncode = 0
        mock_proc.wait = lambda timeout=None: 0
        dummy_out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return mock_proc

    with patch("termux_diffusion.hardware.resolve_device_backend", return_value=("cpu", 0)), \
         patch("termux_diffusion.core.resolve_model_path", return_value=dummy_model), \
         patch("termux_diffusion.core.locate_sd_cli", return_value=Path("/usr/bin/sd-cli")), \
         patch("termux_diffusion.core.check_memory_safety", return_value=(True, "OK")), \
         patch("termux_diffusion.core.subprocess.Popen", side_effect=fake_popen), \
         patch("termux_diffusion.core.export_to_android_gallery", return_value=dummy_out):

        res = generate(
            prompt="korean couple portrait",
            model="z-image-turbo",
            output=dummy_out,
            wake_lock=False,
            export_gallery=False,
            no_cache=True,
        )
        assert "--scheduler" in captured_cmd
        assert "flux" in captured_cmd
        assert "--guidance" in captured_cmd
        assert "3.5" in captured_cmd
        assert res.schedule == "flux"
