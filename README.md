# Termux-Diffusion

[![PyPI](https://img.shields.io/pypi/v/termux-diffusion.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-diffusion/)
[![Python](https://img.shields.io/pypi/pyversions/termux-diffusion.svg?style=flat-square)](https://pypi.org/project/termux-diffusion/)
[![npm](https://img.shields.io/npm/v/termux-diffusion.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-diffusion)
[![npm downloads](https://img.shields.io/npm/dm/termux-diffusion.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-diffusion)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-diffusion)

> **AmfyUI (ComfyUI Mobile DAG Studio) & Sovereign Z-Image Turbo (6.0B DiT) On-Device Diffusion Acceleration for Android Termux.**  
> *Zero PRoot. Zero Virtualization. 100% Native ARM64 Bionic libc, Qualcomm Adreno OpenCL 2.0 / ARM Mali Vulkan 1.1+ GPU Shaders & NEON SIMD Vector Engine.*

---

## 🚀 Major Milestone v2.0.0: AmfyUI & Sovereign 6.0B DiT Execution

`termux-diffusion v2.0.0` introduces **AmfyUI** — an on-device, zero-dependency ComfyUI Directed Acyclic Graph (DAG) visual studio and workflow runtime designed specifically for mobile smartphone viewports. By eliminating heavy desktop Python/PyTorch dependencies (saving over 3.5GB of RAM), AmfyUI compiles and executes ComfyUI JSON workflows directly against Android Bionic libc, Qualcomm Adreno OpenCL, and ARM Mali Vulkan compute kernels.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AmfyUI Mobile Canvas Studio (Port 11553)                        │
├───────────────────────┬───────────────────────────────┬────────────────────────────────┤
│   [CLIPLoader]        │     [CLIPTextEncode]          │       [KSampler (DiT 6.0B)]    │
│  Qwen3-4B-Q2_K.gguf   │   "A stunningly beautiful...  │     z_image_turbo-Q8_0.gguf    │
│   (CPU Big-Cores)     │    Korean woman smiling..."   │     (Adreno OpenCL 2.0)        │
│          │            │               │               │                │               │
│          └────────────┼───────────────┘               │                │               │
│                       ▼                               ▼                │               │
│              [Prompt Conditioning] ──────────────────► [Tiled Flash Attention Kernel]  │
│                                                                        │               │
│                                                                        ▼               │
│   [EmptyLatentImage] ────────────────────────────────────────► [8 Denoising Steps]     │
│    1280 × 720 HD                                                       │               │
│                                                                        ▼               │
│                                                                 [VAEDecode (TAESD)]    │
│                                                                  taef1.safetensors     │
│                                                                        │               │
│                                                                        ▼               │
│                                                                 [SaveImage & Gallery]  │
│                                                                1280x720 TrueColor PNG  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Architectural Nomenclature: What is AmfyUI?

AmfyUI is an engineered recursive backronym and structural design protocol:

* **A — AMEVA / Asynchronous**: Tri-Engine asymmetric pipelining separating Text Encoder (CPU), DiT Backbone (GPU OpenCL/Vulkan), and Latent Decoder (Host CPU).
* **M — Mobile-First Memory**: Strict hardware VRAM quota bounds (`--max-vram`) and dynamic layer streaming preventing Android Low Memory Killer (LMK) aborts.
* **F — Flow-Based DAG**: 100% ComfyUI UI & API Prompt JSON compatibility with topological sorting and acyclic dependency resolution.
* **Y — Yield-Optimized Stream**: Zero-copy mmap tensor paging and AXI bus streaming delivering maximal throughput on thermal-constrained mobile SoCs.
* **UI — User Interface Studio**: Ultra-lightweight zero-dependency web canvas studio optimized for touch gestures and smartphone viewports (360px–430px).

---

## 🧩 ComfyUI Compatibility Matrix & Operational Boundaries

AmfyUI interprets and executes official ComfyUI JSON graph exports directly:

| ComfyUI Node Class | AmfyUI Native Implementation | Hardware Backend | Status |
| :--- | :--- | :--- | :---: |
| **`CLIPLoader`** / **`DualCLIPLoader`** | Qwen3-4B / CLIP ViT-L / T5-XXL (GGUF/FP8) | CPU NEON SIMD (4–8 Threads) | **Native** |
| **`UNETLoader`** / **`DiffusionModelLoader`** | Z-Image Turbo 6.0B DiT, SDXS, SD 1.5, SDXL | Adreno OpenCL / Mali Vulkan / CPU | **Native** |
| **`VAELoader`** / **`TAESDLoader`** | TAESD FLUX.1 (10MB) & SD VAE (FP16) | CPU NEON Vectorized Decode | **Native** |
| **`CLIPTextEncode`** | Prompt Conditioning & Cross-Attention Vectorizer | Host Shared RAM Context | **Native** |
| **`EmptyLatentImage`** | Spatial Latent Allocator (up to 1280×720 HD) | Zero-Copy Buffer Pool | **Native** |
| **`KSampler`** / **`KSamplerAdvanced`** | Res_Multistep, Euler, Euler_A, DPM++ 2M, LCM | Tiled Flash Attention ODE Solver | **Native** |
| **`VAEDecode`** / **`VAEEncode`** | Latent-to-RGB Reconstruction / Img2Img Tiler | Spatial Tiled Decoder (`--vae-tiling`) | **Native** |
| **`SaveImage`** | PNG Export & Android MediaStore Broadcaster | Samsung Gallery Auto-Sync | **Native** |
| **`LoraLoader`** | Runtime Weight Additive Patching | GGML Layer Merging | **Supported** |
| **`Custom Desktop Python Nodes`** | External modules requiring desktop CUDA/GCC | AMEVA-Cluster RPC Offloading | **Cluster Offload** |

---

## 📊 Physical Empirical Benchmark: Galaxy S20 HD 1280×720 4-Season Showcase

> **Jira Ticket: SCRUM-481 | Physical Device: Samsung Galaxy S20 5G**  
> **SoC**: Qualcomm Snapdragon 865 KONA (Cortex-A77 × 4 + A55 × 4)  
> **GPU**: Qualcomm Adreno 650 (OpenCL 2.0 Full Profile) | **RAM**: 12GB LPDDR5  
> **Model**: Tongyi Wanxiang Z-Image Turbo (6.0B DiT) + Qwen3-4B LLM + TAESD  
> **Synthesis Resolution**: **1280 × 720 (16:9 Wide High-Definition)**, 8 steps

| Season & Stage | Model Quant | Hardware Backend | Total Elapsed | Step Latency | Acceleration | Peak Temp | Forensic Sanity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🌸 **Stage 1 (Spring)** | Q4_0 (3.53 GB) | CPU (ARM NEON) | 9,022s (2h 30m) | ~1,180s/it | 1.00× (Base) | 47.8°C | NaN=0 / Zero=0 |
| ☀️ **Stage 2 (Summer)** | Q4_0 (3.53 GB) | Adreno OpenCL 2.0 | **3,835s (1h 03m)** | ~468s/it | **2.35× GPU** | **41.4°C Cool** | NaN=0 / Zero=0 |
| 🍂 **Stage 3 (Autumn)** | Q8_0 (6.13 GB) | CPU (ARM SDOT) | 8,534s (2h 22m) | ~1,060s/it | 1.06× (vs Q4) | 47.1°C | NaN=0 / Zero=0 |
| ❄️ **Stage 4 (Winter)** | Q8_0 (6.13 GB) | Adreno OpenCL 2.0 | **3,895s (1h 04m)** | ~475s/it | **2.19× GPU** | **41.8°C Cool** | NaN=0 / Zero=0 |

### 🔬 Key Engineering Discoveries
1. **2.35× Hardware Acceleration via OpenCL Flash Attention**: GPU compute shaders cut execution time from 2.5 hours down to 1 hour, sustaining ~468 seconds per 1280×720 DiT step.
2. **Cortex-A77 Hardware SDOT Paradox**: On CPU, 8-bit quantization (Q8_0, 8,534s) ran 488 seconds faster than 4-bit (Q4_0, 9,022s) due to ARMv8.2-A hardware `SDOT` executing 4 FMAs per cycle without software bit-shifting unpack loops.
3. **Thermal Superiority**: Adreno OpenCL GPU maintained **41.4°C–41.8°C**, running ~6°C cooler than CPU core saturation (47.8°C).
4. **Complete Forensic Sanity**: 32 distinct layer checkpoints recorded exactly **NaN=0, Zero=0**.

---

## 🖼️ 4-Season 1280×720 HD Portrait Gallery

All images below are native 1280×720 TrueColor PNG artifacts produced during SCRUM-481 on Samsung Galaxy S20:

| 🌸 Spring (Q4 CPU, 1280×720) | ☀️ Summer (Q4 OpenCL, 1280×720) |
| :---: | :---: |
| ![Spring](https://raw.githubusercontent.com/uno-km/termux-diffusion/main/assets/gallery/season1_spring_q4_cpu_1280x720.png) | ![Summer](https://raw.githubusercontent.com/uno-km/termux-diffusion/main/assets/gallery/season2_summer_q4_opencl_1280x720.png) |
| *"A stunningly beautiful 20-year-old Korean woman smiling warmly on a sunny spring day, elegant yellow dress, cherry blossom petals falling, cinematic lighting, 8k"* | *"A stunningly beautiful 20-year-old Korean woman smiling refreshingly on a bright summer beach, stylish airy summer dress, vibrant oceanic breeze, 8k"* |
| **🍂 Autumn (Q8 CPU, 1280×720)** | **❄️ Winter (Q8 OpenCL, 1280×720)** |
| ![Autumn](https://raw.githubusercontent.com/uno-km/termux-diffusion/main/assets/gallery/season3_autumn_q8_cpu_1280x720.png) | ![Winter](https://raw.githubusercontent.com/uno-km/termux-diffusion/main/assets/gallery/season4_winter_q8_opencl_1280x720.png) |
| *"A stunningly beautiful 20-year-old Korean woman smiling peacefully in a scenic autumn park, cozy autumn knit sweater, golden afternoon sunlight, colorful maple foliage"* | *"A stunningly beautiful 20-year-old Korean woman smiling warmly on a crisp winter day, elegant winter coat and scarf, snowy winter atmosphere, 8k"* |

---

## 💻 3-Way Practical Manual: CLI, Python, Node.js

### 1. Installation

```bash
# Python SDK & CLI (PyPI)
pkg update && pkg install -y python clang termux-api
pip install termux-diffusion
termux-diffusion install

# Node.js / TypeScript SDK (NPM)
pkg update && pkg install -y nodejs clang termux-api
npm install -g termux-diffusion
npx termux-diffusion install
```

---

### 2. Terminal CLI & AmfyUI Launch

```bash
# Launch AmfyUI Mobile Studio (Default: host=0.0.0.0, port=11553)
# Automatically binds to 0.0.0.0:11553 (both local and external LAN/Tailscale access)
termux-diffusion ui

# Access from phone: http://localhost:11553
# Access from PC/LAN: http://<phone-ip>:11553 (e.g. http://<phone-ip>:11553)

# Optional: Change port
termux-diffusion ui --port 11554

# Execute a ComfyUI JSON workflow directly
termux-diffusion workflow run workflows/test2_s20_opencl_winter.json \
  --device opencl \
  --output-dir /sdcard/Pictures/TermuxDiffusion

# Validate and inspect workflow node order
termux-diffusion workflow inspect workflows/test1_s20_cpu_spring.json

# High-Level Single Command Synthesis
termux-diffusion generate "A stunningly beautiful Korean woman smiling warmly in winter" \
  --preset z-image-turbo \
  --device opencl \
  -W 1280 -H 720 \
  --steps 8 \
  --diffusion-fa \
  --max-vram GPUOpenCL=2.0 \
  -o /sdcard/Pictures/TermuxDiffusion/winter_hd.png
```

---

### 3. Python SDK Manual (`pip install termux-diffusion`)

```python
import termux_diffusion as td
from termux_diffusion.workflow import WorkflowExecutor

# 1. Execute ComfyUI Workflow JSON Programmatically
executor = WorkflowExecutor()
result_path = executor.execute_file(
    "workflows/test2_s20_opencl_winter.json",
    device="opencl",
    output_dir="/sdcard/Pictures/TermuxDiffusion"
)
print(f"Workflow 1280x720 TrueColor PNG saved to: {result_path}")

# 2. High-Level Python API Synthesis
image_path = td.generate(
    prompt="A stunningly beautiful 20-year-old Korean woman smiling warmly on a crisp winter day, 8k",
    preset="z-image-turbo",
    width=1280,
    height=720,
    steps=8,
    cfg_scale=1.0,
    guidance=3.5,
    device="opencl",
    diffusion_fa=True,
    max_vram="GPUOpenCL=2.0",
    output_path="/sdcard/Pictures/TermuxDiffusion/winter_portrait.png"
)
print(f"Generated via 6.0B DiT & Synced to Samsung Gallery: {image_path}")
```

---

### 4. Node.js / TypeScript SDK Manual (`npm install termux-diffusion`)

```typescript
import { generate } from "termux-diffusion";

async function main(): Promise<void> {
  const result = await generate({
    prompt: "A stunningly beautiful 20-year-old Korean woman smiling warmly on a crisp winter day, 8k",
    preset: "z-image-turbo",
    width: 1280,
    height: 720,
    steps: 8,
    cfgScale: 1.0,
    device: "opencl",
    diffusionFa: true,
    maxVram: "GPUOpenCL=2.0",
    outputPath: "/sdcard/Pictures/TermuxDiffusion/winter_portrait.png"
  });

  console.log("DiT Synthesis Complete. Artifact:", result.outputPath);
}

main().catch(console.error);
```

---

## 📦 Downloadable ComfyUI Workflows & Foundation Links

* **Official AmfyUI Portal & Technical Report**: [https://uno-km.vercel.app/lib/diffusion/amfyui.html](https://uno-km.vercel.app/lib/diffusion/amfyui.html)
* **Spring 1280×720 CPU Workflow JSON**: [test1_s20_cpu_spring.json](https://uno-km.vercel.app/lib/diffusion/workflows/test1_s20_cpu_spring.json)
* **Winter 1280×720 OpenCL Workflow JSON**: [test2_s20_opencl_winter.json](https://uno-km.vercel.app/lib/diffusion/workflows/test2_s20_opencl_winter.json)
* **Standard Z-Image Turbo Q4 DAG JSON**: [user_z_image_turbo_q4.json](https://uno-km.vercel.app/lib/diffusion/workflows/user_z_image_turbo_q4.json)
* **AMEVA Open-Source Foundation Portal**: [https://uno-km.vercel.app/foundation/index.html](https://uno-km.vercel.app/foundation/index.html)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
