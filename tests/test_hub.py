# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Unit tests for termux-diffusion Model Hub, GGUF validator, SHA256, and cache manager."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from termux_diffusion.hub import (
    DEFAULT_PRESETS,
    clear_cache,
    get_cache_dir,
    get_device_model_catalog,
    is_model_cached,
    list_cached_models,
    list_presets,
    register_model,
    set_cache_dir,
    validate_gguf_file,
    verify_file_sha256,
)
from termux_diffusion.exceptions import ModelNotFoundError


@pytest.fixture
def temp_cache():
    tmp = tempfile.mkdtemp(prefix="td_test_cache_")
    set_cache_dir(tmp)
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


def test_default_presets_exist():
    presets = list_presets()
    assert "realistic" in presets
    assert "speed" in presets
    assert "sdxs" in presets
    assert "turbo" in presets
    assert "anime" in presets
    assert presets["realistic"]["size_mb"] > 1000
    assert presets["sdxs"]["size_mb"] == 651


def test_set_and_get_cache_dir(temp_cache):
    assert get_cache_dir() == temp_cache
    assert temp_cache.is_dir()


def test_register_custom_model(temp_cache):
    register_model(
        name="korean-portrait",
        repo_id="custom/korean-portrait-gguf",
        filename="model-q4.gguf",
        alias="korean_portrait.gguf",
        description="Custom Korean portrait model",
        default_steps=12,
        default_cfg=4.5,
        sha256="abcdef1234567890"
    )
    presets = list_presets()
    assert "korean-portrait" in presets
    assert presets["korean-portrait"]["default_steps"] == 12
    assert presets["korean-portrait"]["default_cfg"] == 4.5
    assert presets["korean-portrait"]["sha256"] == "abcdef1234567890"


def test_gguf_validation_and_cached_models_lifecycle(temp_cache):
    # Initially empty
    assert len(list_cached_models(temp_cache)) == 0
    assert not is_model_cached("realistic", temp_cache)

    # 1. Create valid GGUF file (magic 'GGUF' + uint32 version 3)
    import struct
    valid_model = temp_cache / "realistic.gguf"
    valid_content = b"GGUF" + struct.pack("<I", 3) + b"\x00" * 256
    valid_model.write_bytes(valid_content)

    assert validate_gguf_file(valid_model) is True
    assert is_model_cached("realistic", temp_cache)

    # 2. Create invalid model file
    invalid_model = temp_cache / "corrupted.gguf"
    invalid_model.write_bytes(b"BAD_HEADER_DATA_123")
    assert validate_gguf_file(invalid_model) is False

    cached_list = list_cached_models(temp_cache)
    assert len(cached_list) == 2
    
    valid_entry = next(m for m in cached_list if m["name"] == "realistic.gguf")
    assert valid_entry["is_valid_gguf"] is True

    invalid_entry = next(m for m in cached_list if m["name"] == "corrupted.gguf")
    assert invalid_entry["is_valid_gguf"] is False

    # 3. SHA256 integrity verification
    expected_hash = hashlib.sha256(valid_content).hexdigest()
    assert verify_file_sha256(valid_model, expected_hash) is True
    assert verify_file_sha256(valid_model, "wrong_hash_12345") is False

    # 4. Clear cache
    removed = clear_cache(temp_cache, "realistic")
    assert removed == 1
    assert not is_model_cached("realistic", temp_cache)
    assert len(list_cached_models(temp_cache)) == 1  # corrupted still remains
    clear_cache(temp_cache)
    assert len(list_cached_models(temp_cache)) == 0


def test_get_device_model_catalog(temp_cache):
    # Setup simulated models
    (temp_cache / "realistic.gguf").write_bytes(b"GGUF_TEST_1")
    (temp_cache / "taef1.safetensors").write_bytes(b"VAE_TEST_1")
    (temp_cache / "detail_tweaker_lora.safetensors").write_bytes(b"LORA_TEST_1")
    (temp_cache / "qwen2_clip_encoder.gguf").write_bytes(b"CLIP_TEST_1")

    catalog = get_device_model_catalog(temp_cache)
    assert catalog["total"] == 4
    assert len(catalog["checkpoints"]) == 1
    assert catalog["checkpoints"][0]["name"] == "realistic.gguf"
    assert catalog["checkpoints"][0]["category"] == "checkpoint"

    assert len(catalog["vae"]) == 1
    assert catalog["vae"][0]["name"] == "taef1.safetensors"
    assert catalog["vae"][0]["category"] == "vae"

    assert len(catalog["loras"]) == 1
    assert catalog["loras"][0]["name"] == "detail_tweaker_lora.safetensors"
    assert catalog["loras"][0]["category"] == "lora"

    assert len(catalog["text_encoders"]) == 1
    assert catalog["text_encoders"][0]["name"] == "qwen2_clip_encoder.gguf"
    assert catalog["text_encoders"][0]["category"] == "text_encoder"

