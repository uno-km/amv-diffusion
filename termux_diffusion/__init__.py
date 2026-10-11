# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
from ._version import __version__

from .core import (
    GenerationResult,
    ProgressInfo,
    async_generate,
    generate,
    generate_stream,
    get_default_negative_prompt,
    get_quality_guard_negative_prompt,
    set_default_negative_prompt,
    resolve_vram_and_streaming_policy,
)
from .workflow import (
    DAGEngine,
    NODE_REGISTRY,
    WorkflowError,
    WorkflowExecutor,
    WorkflowFormatError,
    WorkflowParser,
    execute_workflow,
)
from .web import (
    DiffusionWebServer,
    run_web_server,
)
from .exceptions import (
    AmevaTermuxError,
    InferenceTimeoutError,
    ModelDownloadError,
    ModelNotFoundError,
    OOMRiskError,
    PlatformNotSupportedError,
    ProvisioningError,
    TermuxDiffusionError,
)

from .hub import (
    DEFAULT_PRESETS,
    clear_cache,
    download_model,
    get_cache_dir,
    get_device_model_catalog,
    is_model_cached,
    list_cached_models,
    list_presets,
    register_model,
    resolve_model_path,
    set_cache_dir,
)
from .engine import (
    DiffusionEngine,
    DiffusionRuntime,
    get_binary_path,
    locate_sd_cli,
    load,
)
from .doctor import run_doctor
from .installer import provision_engine
from .platform import (
    TermuxWakeLock,
    check_memory_safety,
    export_to_android_gallery,
    get_galaxy_gallery_dir,
    get_memory_info,
    get_optimal_thread_count,
    is_android_termux,
    is_arm64,
)
from .hardware import (
    ComputeBackend,
    HardwareProfile,
    TermuxExecutionGuard,
    detect_hardware_profile,
    detect_hardware,
    format_hardware_report,
    get_kernel_physical_memory_gb,
    resolve_device_backend,
    resolve_device,
)
from .downloader import list_models

from .npu import (
    NPUProfile,
    NPUVendor,
    detect_npu_capabilities,
    get_optimal_heterogeneous_pipeline,
)
from .purger import purge_idle_processes, PurgeSummary
from .safety import (
    evaluate_safe_memory_budget,
    SafeMemoryAssessment,
    save_latent_checkpoint,
    load_latent_checkpoint,
)

__author__ = "uno-km (AMEVA Foundation)"
__license__ = "MIT"

__all__ = [
    "__version__",
    "generate",
    "generate_stream",
    "async_generate",
    "GenerationResult",
    "ProgressInfo",
    "download_model",
    "resolve_model_path",
    "register_model",
    "set_cache_dir",
    "get_cache_dir",
    "is_model_cached",
    "list_cached_models",
    "get_device_model_catalog",
    "clear_cache",
    "list_presets",
    "DEFAULT_PRESETS",
    "DiffusionEngine",
    "DiffusionRuntime",
    "load",
    "get_binary_path",
    "locate_sd_cli",
    "provision_engine",
    "run_doctor",
    "is_android_termux",
    "is_arm64",
    "check_memory_safety",
    "get_memory_info",
    "get_optimal_thread_count",
    "get_galaxy_gallery_dir",
    "export_to_android_gallery",
    "TermuxWakeLock",
    "TermuxExecutionGuard",
    "get_kernel_physical_memory_gb",
    "resolve_vram_and_streaming_policy",
    "HardwareProfile",
    "detect_hardware",
    "resolve_device",
    "list_models",
    "AmevaTermuxError",
    "TermuxDiffusionError",
    "PlatformNotSupportedError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "ProvisioningError",
    "OOMRiskError",
    "purge_idle_processes",
    "PurgeSummary",
    "evaluate_safe_memory_budget",
    "SafeMemoryAssessment",
    "save_latent_checkpoint",
    "load_latent_checkpoint",
    "execute_workflow",
    "WorkflowExecutor",
    "WorkflowParser",
    "DAGEngine",
    "NODE_REGISTRY",
    "WorkflowError",
    "WorkflowFormatError",
]


_LAZY_DETAILER = ("detect_faces_on_image", "refine_face_regions")


def __getattr__(name):
    """Face refinement needs numpy and Pillow; load it only when it is used (PEP 562)."""
    if name in _LAZY_DETAILER:
        from . import detailer
        return getattr(detailer, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
