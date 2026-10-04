# Termux-Diffusion (Python)

[![PyPI](https://img.shields.io/pypi/v/termux-diffusion.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-diffusion/)
[![Python](https://img.shields.io/pypi/pyversions/termux-diffusion.svg?style=flat-square)](https://pypi.org/project/termux-diffusion/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-diffusion)

> **AmfyUI (ComfyUI Mobile DAG Studio) & Sovereign Z-Image Turbo (6.0B DiT) On-Device Diffusion Acceleration for Android Termux.**  
> *Zero PRoot. Zero Virtualization. 100% Native ARM64 Bionic libc, Qualcomm Adreno OpenCL 2.0 / ARM Mali Vulkan 1.1+ GPU Shaders & NEON SIMD Vector Engine.*

---

## 🚀 Major Milestone v2.0.0: AmfyUI & Sovereign 6.0B DiT Execution

`termux-diffusion v2.0.0` introduces **AmfyUI** — an on-device, zero-dependency ComfyUI Directed Acyclic Graph (DAG) visual studio and workflow runtime designed specifically for mobile smartphone viewports. By eliminating heavy desktop Python/PyTorch dependencies (saving over 3.5GB of RAM), AmfyUI compiles and executes ComfyUI JSON workflows directly against Android Bionic libc, Qualcomm Adreno OpenCL, and ARM Mali Vulkan compute kernels.

---

## 🏛️ Architectural Nomenclature: What is AmfyUI?

AmfyUI is an engineered recursive backronym and structural design protocol:

* **A — AMEVA / Asynchronous**: Tri-Engine asymmetric pipelining separating Text Encoder (CPU), DiT Backbone (GPU OpenCL/Vulkan), and Latent Decoder (Host CPU).
* **M — Mobile-First Memory**: Strict hardware VRAM quota bounds (`--max-vram`) and dynamic layer streaming preventing Android Low Memory Killer (LMK) aborts.
* **F — Flow-Based DAG**: 100% ComfyUI UI & API Prompt JSON compatibility with topological sorting and acyclic dependency resolution.
* **Y — Yield-Optimized Stream**: Zero-copy mmap tensor paging and AXI bus streaming delivering maximal throughput on thermal-constrained mobile SoCs.
* **UI — User Interface Studio**: Ultra-lightweight zero-dependency web canvas studio optimized for touch gestures and smartphone viewports (360px–430px).

---

## 🧩 ComfyUI Compatibility Matrix & Supported Nodes

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

### 🔬 Key Discoveries
* **2.35× GPU Speedup**: OpenCL Flash Attention cut generation latency from 2.5 hours down to 1 hour.
* **Cortex-A77 SDOT Advantage**: Q8_0 (8,534s) was 488s faster than Q4_0 (9,022s) due to ARMv8.2-A hardware `SDOT` instructions.
* **Thermal Management**: Adreno OpenCL GPU ran at **41.4°C**, ~6°C cooler than CPU saturation (47.8°C).
* **Sanity**: 32 distinct layer checkpoints verified with **NaN=0, Zero=0**.

---

## 💻 Python SDK Installation & Practical User Manual

### 1. Installation & Engine Provisioning
```bash
pkg update && pkg install -y python clang termux-api
pip install termux-diffusion
termux-diffusion install
```

### 2. Python API Practical Examples

#### [Example 1] Programmatic ComfyUI Workflow Execution
```python
from termux_diffusion.workflow import WorkflowExecutor

# Execute ComfyUI DAG JSON directly on Android
executor = WorkflowExecutor()
result_path = executor.execute_file(
    "workflows/test2_s20_opencl_winter.json",
    device="opencl",
    output_dir="/sdcard/Pictures/TermuxDiffusion"
)
print(f"Workflow 1280x720 TrueColor PNG saved to: {result_path}")
```

#### [Example 2] High-Level 6.0B DiT Image Synthesis
```python
import termux_diffusion as td

# Official Z-Image Turbo 6.0B DiT Synthesis
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

#### [Example 3] Launching AmfyUI Mobile Server from Python
```python
from termux_diffusion.web import run_web_server

# Start AmfyUI Mobile Studio Daemon
run_web_server(host="0.0.0.0", port=11553, blocking=True)
```

---

## 🌐 Official Documentation & Ecosystem Links

* **Official AmfyUI Portal & Technical Report**: [https://uno-km.vercel.app/lib/diffusion/amfyui.html](https://uno-km.vercel.app/lib/diffusion/amfyui.html)
* **Spring 1280×720 CPU Workflow JSON**: [test1_s20_cpu_spring.json](https://uno-km.vercel.app/lib/diffusion/workflows/test1_s20_cpu_spring.json)
* **Winter 1280×720 OpenCL Workflow JSON**: [test2_s20_opencl_winter.json](https://uno-km.vercel.app/lib/diffusion/workflows/test2_s20_opencl_winter.json)
* **AMEVA Open-Source Foundation Portal**: [https://uno-km.vercel.app/foundation/index.html](https://uno-km.vercel.app/foundation/index.html)
* **GitHub Repository**: [https://github.com/uno-km/termux-diffusion](https://github.com/uno-km/termux-diffusion)

---

## 📄 License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
