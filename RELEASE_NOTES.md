# v2.1.0 (2026-10-13) — Mobile Canva Studio & SurfaceFlinger Shield

Integrated On-Device Mobile Canva Studio Engine (Magic Eraser, Magic Grab, Magic Expand, Multi-Layer Decomposition), SurfaceFlinger GPU duty-cycle crash shield, safe-by-default memory purger, and unified ameva-runtime GPU HAL routing.

### Key Changes
- Added Mobile Canva Studio Suite: Magic Eraser (magic_eraser), Magic Grab (magic_grab), Magic Expand (magic_expand), and Magic Layers (decompose_layers).
- Implemented SurfaceFlinger Crash Shield with GPU duty-cycle throttling and safe-mode protection.
- Added Safe Memory Purger with automatic OOM safety budget management.
- Unified GPU acceleration dispatch through ameva-runtime HAL.

---

# Release Notes - termux-diffusion v1.8.1

**Release Tag**: `v1.8.1`  
**Distribution Channels**: PyPI (`termux-diffusion`), NPM (`termux-diffusion`), GitHub Releases  
**Target Platform**: Android Termux (ARM64 / aarch64 Bionic)  
**License**: Apache-2.0  

---

## Highlights & Key Architectural Changes

### 1. Unified 5-Backend Standardization
- **Ecosystem Parameter Whitelist**: Aligned CLI options to standard 5-set `["auto", "gpu", "vulkan", "opencl", "cpu"]`.
- **Fail-Fast Policy**: Eliminates ambiguous or misleading parameter formats. Passing unsupported arguments halts immediately with strict diagnostics.

### 2. Zero-Regression Test Suite
- **Comprehensive Unit Testing**: Validated 86 test cases across diffusion pipeline, scheduler presets, prompt caching, and hardware detection (100% Green).
