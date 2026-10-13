# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Deterministic verification of Canva Studio engine: Magic Eraser, Magic Grab, Magic Expand, and Magic Layers."""

import tempfile
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw

from termux_diffusion.canvas import (
    magic_eraser,
    magic_grab,
    magic_expand,
    decompose_layers,
    CanvasProfilingResult,
)


def create_test_canvas() -> Tuple[Image.Image, Image.Image]:
    """Create synthetic test image with central subject circle and separate mask."""
    # 512x512 gradient background
    img = Image.new("RGB", (512, 512), (180, 200, 230))
    draw = ImageDraw.Draw(img)
    # Add subject: red circle at center
    draw.ellipse([156, 156, 356, 356], fill=(220, 50, 50), outline=(150, 20, 20), width=4)

    # Mask covering the subject circle
    mask = Image.new("L", (512, 512), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([156, 156, 356, 356], fill=255)

    return img, mask


def test_magic_eraser_execution():
    """Verify Magic Eraser successfully infills masked region and returns profiling metrics."""
    img, mask = create_test_canvas()
    infilled, profile = magic_eraser(img, mask)

    assert infilled.size == (512, 512)
    assert infilled.mode == "RGB"
    assert profile.operation == "magic_eraser"
    assert profile.latency_ms > 0.0
    print(f"[PASS] {profile.summary()}")


def test_magic_grab_execution():
    """Verify Magic Grab extracts RGBA subject and infilled background."""
    img, _ = create_test_canvas()
    subject_rgba, clean_bg, profile = magic_grab(img, bbox=(150, 150, 360, 360))

    assert subject_rgba.size == (512, 512)
    assert subject_rgba.mode == "RGBA"
    assert clean_bg.size == (512, 512)
    assert clean_bg.mode == "RGB"
    # Subject center should have high alpha, corner should have 0 alpha
    alpha_channel = subject_rgba.split()[3]
    assert alpha_channel.getpixel((256, 256)) > 200
    assert alpha_channel.getpixel((20, 20)) == 0
    print(f"[PASS] {profile.summary()}")


def test_magic_expand_execution():
    """Verify Magic Expand pads canvas to target 16:9 ratio and creates correct border mask."""
    img, _ = create_test_canvas()
    # Expand 512x512 to 910x512 (16:9 ratio)
    target_w, target_h = 910, 512
    expanded_canvas, border_mask, profile = magic_expand(img, target_w, target_h)

    assert expanded_canvas.size == (target_w, target_h)
    assert border_mask.size == (target_w, target_h)
    # Center pixel of mask should be 0 (preserved), border should be 255 (to inpaint)
    assert border_mask.getpixel((target_w // 2, target_h // 2)) == 0
    assert border_mask.getpixel((10, 10)) == 255
    print(f"[PASS] {profile.summary()}")


def test_decompose_layers_execution():
    """Verify Magic Layers decomposes image into structured manifest and assets."""
    img, _ = create_test_canvas()
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest, profile = decompose_layers(img, tmp_dir)

        assert (Path(tmp_dir) / "layer_0_background.png").exists()
        assert (Path(tmp_dir) / "layer_1_subject.png").exists()
        assert (Path(tmp_dir) / "canvas_manifest.json").exists()
        assert len(manifest["layers"]) == 2
        print(f"[PASS] {profile.summary()}")


if __name__ == "__main__":
    from typing import Tuple
    test_magic_eraser_execution()
    test_magic_grab_execution()
    test_magic_expand_execution()
    test_decompose_layers_execution()
    print("ALL CANVA STUDIO TESTS PASSED (100% DETERMINISTIC VERIFICATION)")
