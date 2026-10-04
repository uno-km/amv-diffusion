"""Command-line interface entry points for termux-diffusion."""

import argparse
import sys
from typing import List, Optional

import logging as _logging

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _e:
        _logging.getLogger(__name__).debug("stdout reconfigure failed: %s", _e)
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _e:
        _logging.getLogger(__name__).debug("stderr reconfigure failed: %s", _e)

from .core import generate
from .hub import clear_cache, download_model, list_cached_models, list_presets
from .exceptions import TermuxDiffusionError, ErrorCode, ExitCode
from .engine import DiffusionEngine, get_binary_path
from .doctor import run_doctor
from .installer import provision_engine


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI router for termux-diffusion."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="termux-diffusion",
        description="Production On-Device AI Image Generation for Android Termux & Samsung Galaxy."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate an AI image from a text prompt")
    gen_parser.add_argument("prompt", type=str, help="Text description of image")
    gen_parser.add_argument("-m", "--model", type=str, default="realistic", help="Model preset or .gguf path (default: realistic)")
    gen_parser.add_argument("-n", "--negative", type=str, default=None, help="Negative text prompt guidance")
    gen_parser.add_argument("-d", "--device", type=str, default="cpu", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Computing device: auto, gpu, vulkan, opencl, cpu (default: cpu)")
    gen_parser.add_argument("-b", "--backend", type=str, dest="device", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Alias for --device")
    gen_parser.add_argument("--gpu", action="store_const", const="gpu", dest="device", help="Force GPU hardware acceleration mode")
    gen_parser.add_argument("--cpu", action="store_const", const="cpu", dest="device", help="Force CPU baseline execution mode")
    gen_parser.add_argument("-s", "--steps", type=int, default=None, help="Denoising steps (default determined by preset, e.g. 10)")
    gen_parser.add_argument("-c", "--cfg", type=float, default=None, help="CFG guidance scale")
    gen_parser.add_argument("-g", "--guidance", type=float, default=None, help="Distilled guidance scale (e.g. 3.5 for SDXS)")
    gen_parser.add_argument("-W", "--width", type=int, default=512, help="Image width (default: 512)")
    gen_parser.add_argument("-H", "--height", type=int, default=512, help="Image height (default: 512)")
    gen_parser.add_argument("-t", "--threads", type=int, default=None, help="CPU threads")
    gen_parser.add_argument("--seed", type=int, default=-1, help="RNG seed (-1 for random, 0 to 4294967295)")
    gen_parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    gen_parser.add_argument("--schedule", "--scheduler", dest="schedule", type=str, default=None, help="Noise schedule (flux, karras, exponential, ays, gits, discrete, etc.)")
    gen_parser.add_argument("--flow-shift", type=float, default=None, help="Shift value for Flow models like Flux/Z-Image")
    gen_parser.add_argument("--detail-faces", "--face-detailer", action="store_true", default=False, help="Enable Mobile FaceDetailer (auto face crop, local Img2Img, seamless alpha blend)")
    gen_parser.add_argument("--detailer-strength", type=float, default=0.35, help="FaceDetailer denoising strength (default: 0.35)")
    gen_parser.add_argument("--detailer-steps", type=int, default=2, help="FaceDetailer localized sampling steps (default: 2)")
    gen_parser.add_argument("--detailer-padding", type=float, default=1.45, help="FaceDetailer bounding box padding factor (default: 1.45)")
    gen_parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling for ~70%% lower peak memory")
    gen_parser.add_argument("-i", "--init-img", type=str, default=None, help="Source image for Img2Img synthesis")
    gen_parser.add_argument("--strength", type=float, default=None, help="Img2Img denoising strength (0.0 to 1.0, default: 0.75)")
    gen_parser.add_argument("--lora-dir", type=str, default=None, help="Directory path containing LoRA weights")
    gen_parser.add_argument("--clip-skip", type=int, default=None, help="CLIP layers to skip (1 or 2)")
    gen_parser.add_argument("--control-net", type=str, default=None, help="Path to ControlNet model")
    gen_parser.add_argument("--control-image", type=str, default=None, help="Path to ControlNet guide image")
    gen_parser.add_argument("--control-strength", type=float, default=None, help="ControlNet strength (0.0 to 2.0, default: 0.9)")
    gen_parser.add_argument("--taesd", "--tae", "--fast-vae", dest="taesd", nargs="?", const="auto", type=str, default=None, help="Enable Tiny AutoEncoder (TAESD) for ~1s fast decoding. Pass optional path or 'auto'.")
    gen_parser.add_argument("--strict-vulkan", action="store_true", help="Disallow CPU fallback and fail if Vulkan physical GPU discovery fails")
    gen_parser.add_argument("--preset", type=str, default=None, choices=["fast", "balanced", "anime-experimental", "anime_experimental", "z-image-turbo", "z-image", "turbo-6b", "sdxs"], help="Verified production preset (fast, balanced, anime-experimental, z-image-turbo)")
    gen_parser.add_argument("--llm", type=str, default=None, help="Path or preset for LLM text encoder (e.g. Qwen for Z-Image)")
    gen_parser.add_argument("--diffusion-model", type=str, default=None, help="Path or preset for standalone diffusion model")
    gen_parser.add_argument("--vae", type=str, default=None, help="Path or preset for standalone VAE model")
    gen_parser.add_argument("--clip-l", type=str, default=None, help="Path to standalone CLIP text encoder")
    gen_parser.add_argument("--diffusion-fa", action="store_true", default=False, help="Enable Flash Attention in diffusion model")
    gen_parser.add_argument("--offload-to-cpu", action="store_true", default=False, help="Offload idle modules to CPU RAM to minimize peak VRAM")
    gen_parser.add_argument("--clip-on-cpu", action="store_true", default=False, help="Keep CLIP / LLM text encoder on CPU to preserve GPU VRAM")
    gen_parser.add_argument("--vae-on-cpu", action="store_true", default=False, help="Keep VAE decoder on CPU to preserve GPU VRAM")
    gen_parser.add_argument("--mmap", action="store_true", default=True, help="Memory-map model weights (default: True)")
    gen_parser.add_argument("--no-mmap", action="store_false", dest="mmap", help="Disable memory-mapping")
    gen_parser.add_argument("--max-vram", type=str, default=None, help="Maximum VRAM budget in GiB (e.g. '0' for full in-VRAM residency, '1' or 'vulkan0=1' for 1GiB lifeboat streaming, '1.5' for balanced chunk buffer). Auto-scales deterministically by kernel RAM tier if omitted.")
    gen_parser.add_argument("--stream-layers", action="store_true", default=False, help="Enable layer residency prefetch streaming on top of --max-vram (Tier 3 fallback; disabled in Tier 1/2)")
    gen_parser.add_argument("--params-backend", type=str, default=None, help="Parameter backend assignment (e.g. 'diffusion=cpu' or 'disk')")
    gen_parser.add_argument("--runtime-backend", type=str, default=None, dest="runtime_backend", help="Custom runtime backend assignment (e.g. 'clip=cpu,vae=cpu,diffusion=vulkan0')")
    gen_parser.add_argument("--rpc", type=str, default=None, help="Distributed RPC server addresses (e.g. '192.168.0.220:50052,192.168.0.253:50052')")
    gen_parser.add_argument("-ts", "--tensor-split", type=str, default=None, help="Fraction of the model to offload across devices (e.g. '50,50' or '60,40')")
    gen_parser.add_argument("--vae-format", type=str, default=None, help="VAE latent format override (auto, flux, sd3, flux2, wan)")
    gen_parser.add_argument("--no-cache", "--non-cache", "--no-cached", dest="no_cache", action="store_true", default=False, help="Bypass prompt embedding cache and force full real-time text encoder execution")
    gen_parser.add_argument("--progress", "--show-progress", action="store_true", default=False, help="Display real-time interactive terminal progress bar with step/speed/ETA metrics")
    gen_parser.add_argument("--json-progress", "--progress-json", action="store_true", default=False, help="Stream structured NDJSON progress events for IPC and programmatic consumers")
    gen_parser.add_argument("--progress-file", type=str, default=None, help="Path to write atomic JSON progress snapshots for external monitoring")
    gen_parser.add_argument("--sampling-method", type=str, dest="sampler", help=argparse.SUPPRESS)
    gen_parser.add_argument("--cfg-scale", type=float, dest="cfg", help=argparse.SUPPRESS)
    gen_parser.add_argument("--isolate-text-encoder", "--sequential-stages", dest="isolate_text_encoder", action="store_true", default=False, help="Enable 2-stage memory isolation: execute text encoder independently, kill process to recover RAM, then execute DiT inference")
    gen_parser.add_argument("--safe-mode", action="store_true", default=False, help="Preserve SurfaceFlinger 60/120fps UI and background apps (nice +10, VAE Tiling, 90% GPU duty cycle)")
    gen_parser.add_argument("--keep-apps-alive", action="store_true", default=False, help="Alias for --safe-mode")
    gen_parser.add_argument("--gpu-duty-cycle", type=int, default=None, help="GPU load percentage cap (e.g. 90 for 90% duty cycle, 10% idle gap for SurfaceFlinger)")

    # install command
    inst_parser = subparsers.add_parser("install", help="Provision native Bionic CPU engine (sd-cli-cpu & libomp.so)")
    inst_parser.add_argument("-f", "--force", action="store_true", default=False, help="Force re-installation")
    inst_parser.add_argument("--dedicate", action="store_true", default=False, help="Smart inspection mode: preserve AMEVA runtime symlink, auto-upgrade legacy binaries")

    # doctor command
    subparsers.add_parser("doctor", help="Run 7-tier pre-flight diagnostic checks")

    # models command
    subparsers.add_parser("models", help="List available model presets and locally cached weights")

    # download command
    dl_parser = subparsers.add_parser("download", help="Pre-download a model preset or HF repo")
    dl_parser.add_argument("model", type=str, help="Model preset name (e.g. realistic, speed, sdxs, turbo, anime, fast, balanced)")

    # clear-cache command
    subparsers.add_parser("clear-cache", help="Purge cached model weights to reclaim disk space")

    # workflow command (ComfyUI DAG execution)
    wf_parser = subparsers.add_parser("workflow", help="Execute ComfyUI workflows (UI & API JSON formats)")
    wf_sub = wf_parser.add_subparsers(dest="workflow_command", help="Workflow subcommands")

    wf_run = wf_sub.add_parser("run", help="Execute ComfyUI workflow JSON end-to-end")
    wf_run.add_argument("workflow_file", type=str, help="Path to ComfyUI workflow JSON (UI or API format)")
    wf_run.add_argument("-o", "--output-dir", type=str, default=None, help="Directory to save generated outputs")
    wf_run.add_argument("-d", "--device", type=str, default="auto", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Compute device (default: auto)")
    wf_run.add_argument("--force-run", action="store_true", default=False, help="Bypass hardware limits for high-RAM (16GB/24GB+) devices")
    wf_run.add_argument("--progress", action="store_true", default=True, help="Display progress bar")
    wf_run.add_argument("--json-progress", action="store_true", default=False, help="Stream NDJSON progress events for Node.js / IPC")

    wf_inspect = wf_sub.add_parser("inspect", help="Inspect and validate ComfyUI workflow nodes and execution order")
    wf_inspect.add_argument("workflow_file", type=str, help="Path to ComfyUI workflow JSON")

    # web command (Mobile Web UI server - AmfyUI)
    web_parser = subparsers.add_parser("web", aliases=["ui", "amfy", "amfyui"], help="Launch AmfyUI Mobile Studio & ComfyUI-compatible DAG Visualizer")
    web_parser.add_argument("-p", "--port", type=int, nargs="?", const=11553, default=11553, help="Web server port (default: 11553)")
    web_parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")

    # ── AMEVA Component Protocol v1 ─────────────────────────────────────────
    try:
        from ameva_component.cli_support import build_protocol_subcommands
        build_protocol_subcommands(subparsers)
    except ImportError as _proto_err:
        import logging
        logging.getLogger("termux_diffusion.cli").debug(
            "Optional AMEVA component CLI protocol subcommands omitted: %s", _proto_err
        )
    # ────────────────────────────────────────────────────────────────────────

    if len(argv) == 0:
        parser.print_help()
        return 0

    args, unknown_args = parser.parse_known_args(argv)

    if args.command == "generate":
        if args.seed < -1 or args.seed > 4294967295:
            parser.error("--seed must be between -1 and 4294967295.")

        model_name = args.model
        steps = args.steps
        cfg = args.cfg
        width = args.width
        height = args.height
        sampler = args.sampler
        device = args.device or "cpu"
        vae_tiling = args.vae_tiling

        if args.preset:
            p_key = args.preset.replace("_", "-").lower()
            if p_key in ("z-image", "turbo-6b"):
                p_key = "z-image-turbo"
            from .hub import DEFAULT_PRESETS
            if p_key in DEFAULT_PRESETS:
                p_cfg = DEFAULT_PRESETS[p_key]
                if "alias" in p_cfg and p_cfg["alias"] in DEFAULT_PRESETS:
                    p_cfg = DEFAULT_PRESETS[p_cfg["alias"]]
                model_name = p_cfg.get("alias", p_key)
                if steps is None:
                    steps = p_cfg.get("default_steps")
                if cfg is None:
                    cfg = p_cfg.get("default_cfg")
                if width == 512 and "default_width" in p_cfg:
                    width = p_cfg["default_width"]
                if height == 512 and "default_height" in p_cfg:
                    height = p_cfg["default_height"]
                if sampler is None:
                    sampler = p_cfg.get("default_sampler")
                if device in ("auto", "cpu") and "default_device" in p_cfg:
                    device = p_cfg["default_device"]
                if not vae_tiling and p_cfg.get("default_vae_tiling"):
                    vae_tiling = True
                # Auto-bind DiT multi-component weights
                if not args.diffusion_model and "default_diffusion_model" in p_cfg:
                    args.diffusion_model = p_cfg["default_diffusion_model"]
                elif not args.diffusion_model and p_key == "z-image-turbo":
                    args.diffusion_model = p_cfg.get("filename", "z_image_turbo-Q2_K.gguf")
                if not args.llm and "default_llm" in p_cfg:
                    args.llm = p_cfg["default_llm"]
                if not args.vae and not args.taesd and "default_vae" in p_cfg:
                    args.vae = p_cfg["default_vae"]
                if not args.vae_format and "default_vae_format" in p_cfg:
                    args.vae_format = p_cfg["default_vae_format"]

        try:
            res = generate(
                prompt=args.prompt,
                model=model_name,
                negative_prompt=args.negative,
                device=device,
                steps=steps,
                cfg_scale=cfg,
                guidance=args.guidance,
                width=width,
                height=height,
                threads=args.threads,
                seed=args.seed,
                output=args.output,
                sampling_method=sampler,
                schedule=args.schedule,
                vae_tiling=vae_tiling,
                init_img=args.init_img,
                strength=args.strength,
                lora_dir=args.lora_dir,
                clip_skip=args.clip_skip,
                control_net=args.control_net,
                control_image=args.control_image,
                control_strength=args.control_strength,
                taesd=args.taesd,
                llm=args.llm,
                diffusion_model=args.diffusion_model,
                vae=args.vae,
                clip_l=args.clip_l,
                diffusion_fa=args.diffusion_fa,
                offload_to_cpu=args.offload_to_cpu,
                clip_on_cpu=args.clip_on_cpu,
                vae_on_cpu=args.vae_on_cpu,
                mmap=args.mmap,
                max_vram=args.max_vram,
                stream_layers=args.stream_layers,
                params_backend=args.params_backend,
                custom_backend=args.runtime_backend,
                rpc=args.rpc,
                tensor_split=args.tensor_split,
                vae_format=args.vae_format,
                flow_shift=getattr(args, "flow_shift", None),
                detail_faces=getattr(args, "detail_faces", False),
                detailer_strength=getattr(args, "detailer_strength", 0.35),
                detailer_steps=getattr(args, "detailer_steps", 2),
                detailer_padding=getattr(args, "detailer_padding", 1.45),
                strict_vulkan=args.strict_vulkan,
                no_cache=getattr(args, "no_cache", False),
                progress=getattr(args, "progress", False),
                json_progress=getattr(args, "json_progress", False),
                progress_file=getattr(args, "progress_file", None),
                extra_args=unknown_args if unknown_args else None,
                isolate_text_encoder=getattr(args, "isolate_text_encoder", False),
                safe_mode=getattr(args, "safe_mode", False),
                keep_apps_alive=getattr(args, "keep_apps_alive", False),
                gpu_duty_cycle=getattr(args, "gpu_duty_cycle", None),
                auto_provision=True
            )
            return 0 if res.path.exists() else 1
        except FileNotFoundError as fnf_err:
            print(f"[ERROR] [FILE_NOT_FOUND]: {fnf_err}", file=sys.stderr)
            return ExitCode.CLI_ERROR
        except ValueError as val_err:
            print(f"[ERROR] [INVALID_ARGUMENT]: {val_err}", file=sys.stderr)
            return ExitCode.CLI_ERROR
        except TermuxDiffusionError as td_err:
            print(f"[ERROR] [{td_err.code}]: {td_err}", file=sys.stderr)
            return ExitCode.BUILD_ERROR

    elif args.command == "install":
        provision_engine(force=args.force, dedicate=getattr(args, "dedicate", False))
        return 0

    elif args.command == "doctor":
        ok = run_doctor()
        return 0 if ok else 1

    elif args.command == "models":
        print("\n--- [Presets] Available Presets ---")
        for k, v in list_presets().items():
            desc = v.get('description', f"Alias to {v.get('alias')}")
            size = f" ({v.get('size_mb')}MB)" if v.get('size_mb') else ""
            print(f"  - {k:15} : {desc}{size}")
        print("\n--- [Models] Locally Cached Models ---")
        cached = list_cached_models()
        if not cached:
            print("  (No models cached yet. Run 'termux-diffusion download <model>' or generate)")
        for m in cached:
            print(f"  - {m['name']:25} [{m['size_mb']} MB] -> {m['path']}")
        print()
        return 0

    elif args.command == "download":
        download_model(args.model)
        return 0

    elif args.command == "clear-cache":
        removed = clear_cache()
        print(f"[Clean] Removed {removed} cached model files.")
        return 0

    elif args.command == "workflow":
        from .workflow import WorkflowExecutor, WorkflowParser, DAGEngine, WorkflowError
        if getattr(args, "workflow_command", None) == "run":
            executor = WorkflowExecutor()
            try:
                res = executor.run(
                    workflow=args.workflow_file,
                    output_dir=args.output_dir,
                    device=args.device,
                    force_run=args.force_run,
                    progress=args.progress,
                    json_progress=args.json_progress
                )
                print(f"[Workflow] Completed in {res.elapsed_seconds:.2f}s! Generated {len(res.output_images)} image(s):")
                for img in res.output_images:
                    print(f"  -> {img}")
                return 0 if res.success else 1
            except WorkflowError as wf_err:
                print(f"[ERROR] [WORKFLOW_FAILED]: {wf_err}", file=sys.stderr)
                return 1
            except FileNotFoundError as fnf:
                print(f"[ERROR] [FILE_NOT_FOUND]: {fnf}", file=sys.stderr)
                return 1
        elif getattr(args, "workflow_command", None) == "inspect":
            try:
                nodes = WorkflowParser.parse(args.workflow_file)
                order = DAGEngine.topological_sort(nodes)
                raw_json = WorkflowParser.load_json(args.workflow_file)
                fmt = WorkflowParser.detect_format(raw_json)
                print(f"\n--- [Workflow Inspector] ---")
                print(f"File: {args.workflow_file}")
                print(f"Detected Format: {fmt}")
                print(f"Total Nodes: {len(nodes)}")
                print(f"Execution Order ({len(order)} steps):")
                for idx, nid in enumerate(order, 1):
                    n = nodes[nid]
                    title_str = f" ('{n.title}')" if n.title else ""
                    print(f"  {idx}. [Node #{nid}] {n.class_type}{title_str}")
                print()
                return 0
            except WorkflowError as wf_err:
                print(f"[ERROR] [INSPECT_FAILED]: {wf_err}", file=sys.stderr)
                return 1
            except FileNotFoundError as fnf:
                print(f"[ERROR] [FILE_NOT_FOUND]: {fnf}", file=sys.stderr)
                return 1
        else:
            wf_parser.print_help()
            return 0

    elif args.command in ("web", "ui", "amfy", "amfyui"):
        from .web import run_web_server
        run_web_server(host=args.host, port=args.port, blocking=True)
        return 0

    elif args.command in ("component", "model", "instance"):
        try:
            from ameva_component.cli_support import dispatch_protocol
            from termux_diffusion.control import DiffusionControl
            dispatch_protocol(args, DiffusionControl())
        except ImportError:
            print("[ERROR] ameva-component-sdk not installed.", file=sys.stderr)
            return 1
        return 0

    parser.print_help()
    return 0


def run_install_cli(argv: Optional[List[str]] = None):
    """Entry point for termux-diffusion-install with CLI mutex validation and mode control."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="termux-diffusion-install",
        description="Provision native ARM64 engine binaries with Prebuilt-First pipeline, self-test, and fallback."
    )
    parser.add_argument("--prebuilt", action="store_true", help="Install pre-compiled binary (prebuilt-only mode)")
    parser.add_argument("--prebuilt-only", action="store_true", help="Install pre-compiled binary (do not fall back to source build on failure)")
    parser.add_argument("--build-from-source", action="store_true", help="Skip prebuilt binary and compile from source")
    parser.add_argument("-b", "--backend", type=str, default="cpu", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Target compute backend (default: cpu)")
    parser.add_argument("-f", "--force", "--force-reinstall", dest="force", action="store_true", help="Force re-installation")
    parser.add_argument("--print-diagnostics", action="store_true", help="Print system, ABI, CPU, and Vulkan diagnostic information and continue")
    parser.add_argument("--diagnostics-only", action="store_true", help="Print system, ABI, CPU, and Vulkan diagnostic information and exit immediately")

    args = parser.parse_args(argv)

    # Mutually Exclusive Flag Check
    is_prebuilt_flag = args.prebuilt or args.prebuilt_only
    if is_prebuilt_flag and args.build_from_source:
        print("[ERROR] E_CLI_EXCLUSIVE_MUTEX: --prebuilt / --prebuilt-only and --build-from-source are mutually exclusive options.", file=sys.stderr)
        sys.exit(ExitCode.CLI_ERROR)

    if is_prebuilt_flag:
        install_mode = "prebuilt-only"
    elif args.build_from_source:
        install_mode = "source-only"
    else:
        install_mode = "prebuilt-first"

    print(f"[termux-diffusion] Selected Install Mode: {install_mode}")

    if getattr(args, "print_diagnostics", False) or getattr(args, "diagnostics_only", False):
        print("\n=== [termux-diffusion] System Diagnostics ===")
        from .hardware import detect_hardware_profile, format_hardware_report
        hw = detect_hardware_profile()
        print(format_hardware_report(hw))
        print(f"Install Mode: {install_mode}")
        print(f"Target Backend: {args.backend}")
        print("=============================================\n")
        if getattr(args, "diagnostics_only", False):
            return ExitCode.SUCCESS

    try:
        provision_engine(force=args.force, backend=args.backend, install_mode=install_mode)
    except TermuxDiffusionError as err:
        print(f"[ERROR] [{err.code}]: {err}", file=sys.stderr)
        if err.code == ErrorCode.MANIFEST_DOWNLOAD or "E_PREBUILT_FAILED" in str(err) or err.code == "E_PREBUILT_UNAVAILABLE":
            sys.exit(ExitCode.INTEGRITY_ERROR)
        elif err.code == ErrorCode.INSTALL_LOCKED:
            sys.exit(ExitCode.CLI_ERROR)
        else:
            sys.exit(ExitCode.BUILD_ERROR)


def run_doctor_cli():
    """Entry point for termux-diffusion-doctor."""
    run_doctor()


if __name__ == "__main__":
    sys.exit(main())
