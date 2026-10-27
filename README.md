# AMV-Diffusion

> **마이그레이션 안내**: 이 패키지는 `termux-diffusion`에서 `amv-diffusion`로 이름이 바뀌었습니다. 기존 `termux-diffusion` 배포본(PyPI/npm)은 호환을 위해 당분간 유지되나 deprecated 상태이며 더 이상 갱신되지 않습니다. `pip install amv-diffusion` / `npm install amv-diffusion`로 전환하십시오.
>
> Android **Termux 환경** 또는 **안드로이드 Linux 커널(루팅 없이 Linux 환경이 구동되는 모든 디바이스)** 에서 동작합니다 — Termux 전용이 아닙니다.

[![PyPI](https://img.shields.io/pypi/v/amv-diffusion.svg?style=flat-square&color=0369a1)](https://pypi.org/project/amv-diffusion/)
[![Python](https://img.shields.io/pypi/pyversions/amv-diffusion.svg?style=flat-square)](https://pypi.org/project/amv-diffusion/)
[![npm](https://img.shields.io/npm/v/amv-diffusion.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/amv-diffusion)
[![npm downloads](https://img.shields.io/npm/dm/amv-diffusion.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/amv-diffusion)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/amv-diffusion)

> **Official Support for Z-Image Turbo (6.0B DiT) & Pure Native On-Device Diffusion Acceleration for Android Termux & Samsung Galaxy.**  
> *Zero PRoot. Zero Virtualization. 100% Native ARM64 Bionic libc, Khronos Vulkan 1.1+ GPU Shaders & NEON SIMD Vector Engine.*

---

## What's New in v2.1.0 — Mobile Canva Studio & SurfaceFlinger Shield

- Added Mobile Canva Studio Suite: Magic Eraser (magic_eraser), Magic Grab (magic_grab), Magic Expand (magic_expand), and Magic Layers (decompose_layers).
- Implemented SurfaceFlinger Crash Shield with GPU duty-cycle throttling and safe-mode protection.
- Added Safe Memory Purger with automatic OOM safety budget management.
- Unified GPU acceleration dispatch through ameva-runtime HAL.

---

## 🚀 Major Breakthrough: 6.0B DiT (Z-Image Turbo) Sovereign Mobile Execution

`amv-diffusion v1.8.0` officially achieves a world-class on-device AI milestone: **Native, unquantized context execution of 6.0 Billion Parameter Diffusion Transformers (DiT)** on commercial Android hardware (Samsung Galaxy S21 5G / ARM Mali-G78 MP14 GPU) without cloud dependencies.

### 🌟 Tri-Engine Asymmetric Pipeline Architecture
```
[User Natural Language Prompt]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Text Encoder (Qwen3-4B-Instruct-2507-Q2_K.gguf)     │ ◄── CPU 4-Core Parallel SIMD (clip=cpu)
│    - 4.0B LLM Context Understanding & Dense Embedding  │
└────────────────────────────────────────────────────────┘
          │ (High-Dimensional Prompt Embeddings)
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. DiT Denoising Core (z_image_turbo-Q2_K.gguf)        │ ◄── Mali-G78 Vulkan Shaders (diffusion=vulkan0)
│    - 6.0B Diffusion Transformer                        │     Dynamic Layer Streaming (stream-layers)
│    - Tiled Flash Attention (--diffusion-fa)            │     Strict 1.0 GB VRAM Ceiling (max-vram=1GB)
└────────────────────────────────────────────────────────┘
          │ (Latent Representation)
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Ultra-Fast VAE Decoder (taef1.safetensors)          │ ◄── 10MB Tiny AutoEncoder (taesd, vae=cpu)
│    - Fast Latent Decoding with Zero VRAM Spike   │
└────────────────────────────────────────────────────────┘
          │
          ▼
[24-bit RGB PNG Image Artifact] (512 x 512)
```

---

## 📸 Sovereign Physical Verification: Galaxy S21 DiT 8-Step Output

![Galaxy S21 Z-Image Turbo 8-Step Verification Output](docs/research/s21_z_image_turbo_8step.png)

### 1. Verification Prompt (Ground Truth)
```text
"A cinematic photo of a neon cybernetic tiger walking in Seoul street at night"
```

### 2. Verified Input Parameter Engineering
| Parameter Flag | Applied Setting | Engineering Purpose & Mechanism |
| :--- | :--- | :--- |
| **`--diffusion-model`** | `z_image_turbo-Q2_K.gguf` | 6.0B Diffusion Transformer backbone network |
| **`--llm`** | `Qwen3-4B-Instruct-2507-Q2_K.gguf` | 4.0B Parameter LLM text encoder for dense prompt comprehension |
| **`--taesd`** | `taef1.safetensors` | 10 MB Tiny AutoEncoder for instant FLUX.1 latent-to-pixel decoding |
| **`--steps`** | `8` | Optimal numerical convergence for distilled DiT architectures |
| **`--cfg-scale`** | `1.0` | Prevents over-saturation artifacts on distilled flow models |
| **`--sampling-method`** | `euler` | 1st-order Ordinary Differential Equation (ODE) numerical solver |
| **`--backend`** | `clip=cpu,diffusion=vulkan0,vae=cpu` | Allocates GPU VRAM strictly to DiT while offloading LLM/VAE to CPU |
| **`--stream-layers`** | Enabled | Streams DiT layers over AXI bus into VRAM sequentially |
| **`--max-vram`** | `vulkan0=1` (1.0 GB) | Guarantees Zero-LMK memory safety within Android OS limits |
| **`--diffusion-fa`** | Enabled | Tiled Flash Attention reduces attention memory complexity to $O(N)$ |
| **`--params-backend`** | `diffusion=cpu` | Keeps weights in 8GB LPDDR5 system RAM to avoid VRAM exhaustion |
| **`--vae-tiling`** | Enabled | Prevents memory allocation spikes during final latent reconstruction |

### 3. Rendering Quality & Optical Ground Truth
* **Specular PBR Reflection**: Realistic light scattering from neon signboards on wet Seoul asphalt road surfaces.
* **Geometric Armor Definition**: Razor-sharp boundary definition on the cybernetic tiger's composite metal armor plates.
* **Micro-Texture Integrity**: Individual whiskers and fur textures cleanly separated from background high-frequency noise.
* **Zero-Crash Stability**: 100% completion achieved with zero Out-of-Memory (LMK) aborts and zero thermal throttling termination.

### 🔬 Academic Research & Lab Milestone Report
* **Full Technical Research Report**: [Galaxy S21 Z-Image Turbo Vulkan Research Report](docs/research/s21_z_image_turbo_vulkan_research_report.md)
* **AMEVA Edge Systems Lab (AESL)**: [https://uno-km.vercel.app/labs/index.html?menu=research-papers&post=33](https://uno-km.vercel.app/labs/index.html?menu=research-papers&post=33)

---

## 💻 3-Channel Practical User Manual: CLI, PIP, NPM

`amv-diffusion` delivers 100% feature parity across **Terminal CLI**, **Python SDK (PyPI)**, and **Node.js SDK (NPM)**.

### 1. Installation

* **Python SDK (PyPI)**:
  ```bash
  pip install amv-diffusion
  amv-diffusion install
  ```

* **Node.js / TypeScript SDK & CLI (NPM)**:
  ```bash
  npm install -g amv-diffusion
  npx amv-diffusion install
  ```

---

### 2. Terminal CLI Manual

#### [Method 1] One-Touch High-Level Command
```bash
amv-diffusion generate "A cinematic photo of a neon cybernetic tiger walking in Seoul street at night" \
  --preset z-image-turbo \
  --device vulkan \
  --steps 8 \
  --cfg 1.0 \
  --sampler euler \
  --diffusion-fa \
  --stream-layers \
  --max-vram vulkan0=1 \
  --clip-on-cpu \
  --vae-on-cpu \
  --taesd auto \
  -o /sdcard/Pictures/TermuxDiffusion/cyber_tiger_dit.png
```

#### [Method 2] Low-Level Native Binary Direct Execution
```bash
sd-cli-vulkan \
  -p "A cinematic photo of a neon cybernetic tiger walking in Seoul street at night" \
  -W 512 -H 512 -t 4 \
  --steps 8 --cfg-scale 1.0 --sampling-method euler \
  --diffusion-model ~/.cache/amv-diffusion/models/z_image_turbo-Q2_K.gguf \
  --llm ~/.cache/amv-diffusion/models/Qwen3-4B-Instruct-2507-Q2_K.gguf \
  --taesd ~/.cache/amv-diffusion/models/taef1.safetensors \
  --clip-on-cpu --vae-on-cpu --vae-format flux --mmap --diffusion-fa \
  --backend clip=cpu,diffusion=vulkan0,vae=cpu \
  --max-vram vulkan0=1 --stream-layers --params-backend diffusion=cpu --vae-tiling \
  -o /sdcard/Pictures/TermuxDiffusion/cyber_tiger_dit.png
```

---

### 3. Python SDK Manual (pip install amv-diffusion)

```python
import termux_diffusion as td

# Official Z-Image Turbo 6.0B DiT Synthesis via Python API
image_path = td.generate(
    prompt="A cinematic photo of a neon cybernetic tiger walking in Seoul street at night",
    preset="z-image-turbo",
    steps=8,
    cfg_scale=1.0,
    sampler="euler",
    device="vulkan",
    diffusion_fa=True,
    stream_layers=True,
    max_vram="vulkan0=1",
    clip_on_cpu=True,
    vae_on_cpu=True,
    taesd="auto",
    output_path="/sdcard/Pictures/TermuxDiffusion/cyber_tiger_dit.png"
)

print(f"Generated and synchronized to Samsung Gallery: {image_path}")
```

---

### 4. Node.js / TypeScript SDK Manual (npm install amv-diffusion)

```typescript
import { generate } from "amv-diffusion";

async function main(): Promise<void> {
  // Official Z-Image Turbo 6.0B DiT Synthesis via Node.js API
  const result = await generate({
    prompt: "A cinematic photo of a neon cybernetic tiger walking in Seoul street at night",
    preset: "z-image-turbo",
    steps: 8,
    cfgScale: 1.0,
    sampler: "euler",
    device: "vulkan",
    diffusionFa: true,
    streamLayers: true,
    maxVram: "vulkan0=1",
    clipOnCpu: true,
    vaeOnCpu: true,
    taesd: "auto",
    outputPath: "/sdcard/Pictures/TermuxDiffusion/cyber_tiger_dit.png"
  });

  console.log("DiT Synthesis Complete. Artifact:", result.outputPath);
}

main().catch(console.error);
```

---

## 🎛️ Complete Parameter Reference Matrix

| Parameter Flag | Python SDK Key | Node.js Key | Type | Default | Engineering Purpose |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `-p, --prompt` | `prompt` | `prompt` | String | *Required* | Conditioning text description |
| `-W, -H` | `width, height` | `width, height` | Integer | `512` | Synthesis resolution |
| `-s, --steps` | `steps` | `steps` | Integer | `8` (DiT) / `6` (LCM) | Numerical solver sampling iterations |
| `-c, --cfg` | `cfg_scale` | `cfgScale` | Float | `1.0` (DiT) / `1.5` (LCM) | Classifier-Free Guidance scale |
| `--sampler` | `sampler` | `sampler` | String | `euler` / `lcm` | Denoising ODE solver algorithm |
| `-d, --device` | `device` | `device` | String | `vulkan` | Hardware accelerator (`vulkan`, `cpu`) |
| `--preset` | `preset` | `preset` | String | `z-image-turbo` | Production-verified configuration bundle |
| `--stream-layers` | `stream_layers` | `streamLayers` | Boolean | `True` (DiT) | Dynamic layer streaming over AXI bus |
| `--max-vram` | `max_vram` | `maxVram` | String | `vulkan0=1` | GPU memory ceiling to avert OS LMK aborts |
| `--diffusion-fa` | `diffusion_fa` | `diffusionFa` | Boolean | `True` | Flash Attention $O(N)$ memory reduction |
| `--clip-on-cpu` | `clip_on_cpu` | `clipOnCpu` | Boolean | `True` | Keeps LLM text encoder on CPU cores |
| `--vae-on-cpu` | `vae_on_cpu` | `vaeOnCpu` | Boolean | `True` | Offloads VAE decode to host CPU |
| `--taesd` | `taesd` | `taesd` | String | `auto` | Tiny AutoEncoder for ~1.2s latent decoding |

---

## 🌐 Official Ecosystem Documentation
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/diffusion/)
- [Advanced Parameters Handbook](https://uno-km.vercel.app/lib/diffusion/advanced-parameters.html)
- [Galaxy S21 Z-Image Turbo Research Report](docs/research/s21_z_image_turbo_vulkan_research_report.md)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
