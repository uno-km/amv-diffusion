# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Galaxy S20 Z-Image Turbo 3-Tier Golden Benchmark (Korean Handsome Man Spring Edition)
Derived from Four-Seasons Masterpiece Architecture & User ComfyUI Workflow Settings
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
logger = logging.getLogger("s20_spring_man_benchmark")

# Lower process priority to nice(+10) for SurfaceFlinger UI and background apps preservation
if hasattr(os, "nice") and sys.platform != "win32":
    try:
        os.nice(10)
        logger.info("Process priority lowered to nice(+10) for OS UI and background apps safety")
    except Exception as e:
        logger.warning("os.nice failed: %s", e)

# 1. Base Male Spring Prompt (사계절 여성 버전 -> 남성 버전 고품질 정밀 이식)
PROMPT_BASE = (
    "A remarkably handsome 20-year-old Korean man smiling warmly on a bright spring day, "
    "exquisite sculpted facial features, highly detailed handsome face, defined jawline and clean skin texture, "
    "wearing an elegant light cream knit sweater and pastel jacket, soft spring sunlight filtering through trees, "
    "romantic fluttering cherry blossom petals background, photorealistic portrait, 8k, 35mm film photography, "
    "masterpiece, highly detailed eyes and hair"
)

# 2. High Tier Prompt (사계절보다 더 또렷하고 깊이 있는 눈동자, 짙고 화려한 배경 대비)
PROMPT_HIGH = (
    "A remarkably handsome 20-year-old Korean man smiling warmly on a bright spring day, "
    "exquisite sculpted facial features, highly detailed handsome face, sharp focus on eyes and eyelashes, "
    "defined jawline, wearing an elegant dark navy tailored blazer over cream shirt, vivid spring sunlight "
    "with rich deep contrast, vibrant pink blooming cherry blossom trees, rich saturated background colors, "
    "atmospheric chiaroscuro bokeh, photorealistic portrait, 8k, sharp micro-textures, masterpiece"
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

OUTPUT_DIR = Path.home() / "korean_man_spring_benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIERS = {
    "fast": {
        "title": "Tier 1: Fast (4-Step, 768x768)",
        "prompt": PROMPT_BASE,
        "steps": 4,
        "width": 768,
        "height": 768,
        "output_name": "spring_man_tier1_fast_4s.png",
        "description": "패스트 검증선 (품질 살짝 아쉽지만 이목구비 온전한 합격선)"
    },
    "default": {
        "title": "Tier 2: Default (8-Step, 1024x1024)",
        "prompt": PROMPT_BASE,
        "steps": 8,
        "width": 1024,
        "height": 1024,
        "output_name": "spring_man_tier2_default_8s.png",
        "description": "사계절 명작 기준선 (봄여름가을겨울 원본 퀄리티 100% 복원)"
    },
    "high": {
        "title": "Tier 3: High (14-Step, 1024x1024)",
        "prompt": PROMPT_HIGH,
        "steps": 14,
        "width": 1024,
        "height": 1024,
        "output_name": "spring_man_tier3_high_14s.png",
        "description": "사계절 초월 마스터피스 (더 또렷한 안면, 짙은 벚꽃 배경 대비)"
    }
}


def run_tier(tier_key: str):
    from termux_diffusion.core import generate

    cfg = TIERS[tier_key]
    out_path = OUTPUT_DIR / cfg["output_name"]

    logger.info("=" * 75)
    logger.info("LAUNCHING %s", cfg["title"])
    logger.info("Description: %s", cfg["description"])
    logger.info("Steps: %d, Resolution: %dx%d, Seed: %d", cfg["steps"], cfg["width"], cfg["height"], SEED)
    logger.info("DiT Model: %s", DIT_MODEL)
    logger.info("VAE: %s (No Tiny VAE)", VAE_MODEL)
    logger.info("Output Target: %s", out_path)
    logger.info("=" * 75)

    t0 = time.time()
    res = generate(
        prompt=cfg["prompt"],
        negative_prompt=None,  # ConditioningZeroOut
        model=DIT_MODEL,
        diffusion_model=DIT_MODEL,
        llm=LLM_MODEL,
        vae=VAE_MODEL,
        taesd=None,            # 10MB Tiny VAE 결함 배제
        steps=cfg["steps"],
        cfg_scale=CFG_SCALE,
        width=cfg["width"],
        height=cfg["height"],
        seed=SEED,
        sampling_method=SAMPLER,
        schedule=SCHEDULER,
        flow_shift=FLOW_SHIFT,
        output=str(out_path),
        # 12중 무중단 생존 방어 옵션
        params_backend="diffusion=cpu",
        stream_layers=True,
        max_vram="vulkan0=1",
        vae_tiling=True,
        threads=4,
        wake_lock=True,
        progress=True
    )
    elapsed = time.time() - t0

    size_kb = round(os.path.getsize(res.path) / 1024, 1) if os.path.exists(res.path) else 0
    logger.info("=" * 75)
    logger.info("FINISHED %s in %.2fs (%.2f min)", cfg["title"], elapsed, elapsed / 60)
    logger.info("Artifact Saved: %s (Size: %s KB)", res.path, size_kb)
    logger.info("=" * 75)

    record = {
        "tier": tier_key,
        "title": cfg["title"],
        "steps": cfg["steps"],
        "width": cfg["width"],
        "height": cfg["height"],
        "elapsed_sec": round(elapsed, 2),
        "elapsed_min": round(elapsed / 60, 2),
        "artifact": str(res.path),
        "size_kb": size_kb,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Galaxy S20 Spring Man 3-Tier Runner")
    parser.add_argument("--tier", choices=["fast", "default", "high", "all"], default="all")
    args = parser.parse_args()

    results = []
    if args.tier == "all":
        for t in ["fast", "default", "high"]:
            res = run_tier(t)
            results.append(res)
    else:
        res = run_tier(args.tier)
        results.append(res)

    summary_file = OUTPUT_DIR / f"summary_spring_man_{int(time.time())}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("All tiers completed! Summary report: %s", summary_file)


if __name__ == "__main__":
    main()
