# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Galaxy S20 Z-Image Turbo 3-Tier Golden Preset Benchmark Runner
Derived from user workflow: '이건 되나.json' (Black and White Butterfly Fashion Portrait)
Standards: Zero-Silent-Fallback, Anti-Kill 12-Layer Defense Protocol
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("s20_butterfly_benchmark")

PROMPT = (
    "Dramatic black and white high fashion studio portrait, close-up bust shot, "
    "pale platinum blonde woman with sleek low ponytail, head tilted upward, eyes softly closed, "
    "wearing a fitted black turtleneck top. A large translucent pale white butterfly hovers gently "
    "right at her lips, delicate detailed wing veins visible. Hard rim light creates glowing bright "
    "white halo around her hair and face, deep inky pure black minimalist background, stark high "
    "contrast chiaroscuro lighting, film grain texture, moody ethereal atmosphere, monochrome, "
    "editorial fashion photography, shot on 35mm film, soft subtle skin texture, sharp focus on "
    "butterfly and facial profile, vertical composition, minimalist dark aesthetic, artistic surreal fashion"
)

SEED = 121725701057393
FLOW_SHIFT = 3.0
SAMPLER = "res_multistep"
SCHEDULER = "simple"
CFG_SCALE = 1.0

MODELS_DIR = Path.home() / ".cache" / "termux-diffusion" / "models"
DIT_MODEL = str(MODELS_DIR / "z_image_turbo-Q4_0.gguf")
LLM_MODEL = str(MODELS_DIR / "Qwen3-4B-Instruct-2507-Q2_K.gguf")
VAE_MODEL = str(MODELS_DIR / "z_image_ae.safetensors")

OUTPUT_DIR = Path.home() / "butterfly_benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = {
    "fast": {
        "title": "Tier 1: Fast (4-Step, 768x768)",
        "steps": 4,
        "width": 768,
        "height": 768,
        "output_name": "butterfly_tier1_fast_4s_768.png",
        "rationale": "초고속 형태 및 구도 검증 (품질 살짝 아쉽지만 실용적 완성선)"
    },
    "default": {
        "title": "Tier 2: Default (8-Step, 1024x1024) [이건 되나.json 원본]",
        "steps": 8,
        "width": 1024,
        "height": 1024,
        "output_name": "butterfly_tier2_default_8s_1024.png",
        "rationale": "사계절 명작 기준선 (사용자 워크플로우 100% 동일 복원, 정규 VAE)"
    },
    "high": {
        "title": "Tier 3: High (12-Step, 1024x1024)",
        "steps": 12,
        "width": 1024,
        "height": 1024,
        "output_name": "butterfly_tier3_high_12s_1024.png",
        "rationale": "사계절을 뛰어넘는 고정밀 실사 (나비 날개 시맥 및 피부 잔주름 극대화)"
    }
}


def run_tier(tier_key: str):
    from termux_diffusion.core import generate

    cfg = TIERS[tier_key]
    out_path = OUTPUT_DIR / cfg["output_name"]

    logger.info("=" * 70)
    logger.info("STARTING BENCHMARK: %s", cfg["title"])
    logger.info("Steps: %d, Resolution: %dx%d, Seed: %d", cfg["steps"], cfg["width"], cfg["height"], SEED)
    logger.info("DiT Model: %s", DIT_MODEL)
    logger.info("VAE Model: %s (No Tiny VAE)", VAE_MODEL)
    logger.info("Output: %s", out_path)
    logger.info("=" * 70)

    t0 = time.time()
    res = generate(
        prompt=PROMPT,
        negative_prompt=None,  # Zero-Out
        model=DIT_MODEL,
        diffusion_model=DIT_MODEL,
        llm=LLM_MODEL,
        vae=VAE_MODEL,
        taesd=None,            # Tiny VAE 결함 배제
        steps=cfg["steps"],
        cfg_scale=CFG_SCALE,
        width=cfg["width"],
        height=cfg["height"],
        seed=SEED,
        sampling_method=SAMPLER,
        schedule=SCHEDULER,
        flow_shift=FLOW_SHIFT,
        output=str(out_path),
        # 12중 무중단 방어 옵션
        params_backend="diffusion=cpu",
        stream_layers=True,
        max_vram="vulkan0=1",
        vae_tiling=True,
        safe_mode=True,
        keep_apps_alive=True,
        threads=4,
        wake_lock=True,
        progress=True
    )
    elapsed = time.time() - t0

    size_kb = round(os.path.getsize(res.path) / 1024, 1) if os.path.exists(res.path) else 0
    logger.info("=" * 70)
    logger.info("COMPLETED: %s in %.2fs (%.2fm)", cfg["title"], elapsed, elapsed / 60)
    logger.info("Artifact: %s (%s KB)", res.path, size_kb)
    logger.info("=" * 70)
    return {
        "tier": tier_key,
        "elapsed_sec": round(elapsed, 2),
        "artifact": str(res.path),
        "size_kb": size_kb
    }


def main():
    parser = argparse.ArgumentParser(description="Galaxy S20 Butterfly 3-Tier Runner")
    parser.add_argument("--tier", choices=["fast", "default", "high", "all"], default="default")
    args = parser.parse_args()

    results = []
    if args.tier == "all":
        for t in ["fast", "default", "high"]:
            res = run_tier(t)
            results.append(res)
    else:
        res = run_tier(args.tier)
        results.append(res)

    summary_file = OUTPUT_DIR / f"summary_{int(time.time())}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark summary saved: %s", summary_file)


if __name__ == "__main__":
    main()
