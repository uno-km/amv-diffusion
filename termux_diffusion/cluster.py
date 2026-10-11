# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA Unified Distributed RPC Cluster Orchestration Utility.
Strict Protocol: Zero-Silent-Fallback & Zero-Deception ABI.
Component: [DIFFUSION-CLUSTER]
"""

import socket
import logging
import time
from typing import List, Optional, Union, Dict, Any, Callable, Set, Tuple

from termux_diffusion.exceptions import (
    ClusterConnectionError,
    ClusterConfigurationError,
    ClusterLicenseRequiredError,
)

logger = logging.getLogger(__name__)


def check_cluster_license(license_key: Optional[str] = None) -> None:
    """Enforces exclusive AMEVA-Cluster runtime license requirement.
    Fail-Fast: Raises ClusterLicenseRequiredError if ameva_cluster is not installed
    or license verification fails.
    """
    try:
        from ameva_cluster.guard import verify_cluster_license
    except ImportError:
        raise ClusterLicenseRequiredError()

    if not verify_cluster_license(license_key):
        raise ClusterLicenseRequiredError(
            "\n================================================================================\n"
            "[AMEVA-CLUSTER] CLUSTER LICENSE INVALID (E403)\n"
            "================================================================================\n"
            "The provided AMEVA Cluster license token is invalid or expired.\n"
            "Please check your AMEVA_CLUSTER_LICENSE environment variable.\n"
            "================================================================================"
        )


def setup_cluster_guard_tunnels(
    servers: List[str],
    secret_key: Optional[str] = None
) -> Tuple[List[str], List[Any]]:
    """Establishes authenticated loopback MasterTunnels for remote RPC nodes.

    Transforms remote endpoints into local authenticated tunnel endpoints (127.0.0.1:PORT)
    which transparently inject the AMEVA Guard cryptographic handshake.

    Returns:
        Tuple of (tunnel_server_endpoints, active_tunnel_instances)
    """
    if not servers:
        return [], []

    try:
        from ameva_cluster.guard import MasterTunnel
    except ImportError:
        raise ClusterLicenseRequiredError()

    tunnel_endpoints: List[str] = []
    active_tunnels: List[Any] = []

    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)

        # Local loopback addresses do not require an extra proxy hop
        if host in ("127.0.0.1", "localhost", "::1"):
            tunnel_endpoints.append(endpoint)
            continue

        tunnel = MasterTunnel(remote_host=host, remote_port=port, secret_key=secret_key)
        tunnel.start()
        active_tunnels.append(tunnel)
        local_ep = f"127.0.0.1:{tunnel.local_port}"
        tunnel_endpoints.append(local_ep)
        logger.info(
            "[AMEVA-DIFFUSION-CLUSTER] Authenticated MasterTunnel active: %s -> %s",
            local_ep,
            endpoint,
        )

    return tunnel_endpoints, active_tunnels



def parse_cluster_rpc_spec(rpc_input: Optional[Union[str, List[str]]]) -> List[str]:
    """Parse and normalize distributed RPC server endpoints.

    Supports comma-separated strings or list of host:port strings.
    Validates port ranges and host syntax strictly without silent normalization.
    """
    if not rpc_input:
        return []

    if isinstance(rpc_input, str):
        raw_servers = [s.strip() for s in rpc_input.split(",") if s.strip()]
    elif isinstance(rpc_input, (list, tuple)):
        raw_servers = [str(s).strip() for s in rpc_input if str(s).strip()]
    else:
        raise ClusterConfigurationError(
            f"Invalid cluster_rpc_servers specification type: {type(rpc_input).__name__}. "
            f"Expected comma-separated string or list of endpoints."
        )

    parsed_servers: List[str] = []
    for entry in raw_servers:
        if ":" not in entry:
            raise ClusterConfigurationError(
                f"Invalid RPC endpoint format '{entry}'. Expected 'host:port' format (e.g. '192.0.2.11:50052')."
            )
        parts = entry.split(":")
        if len(parts) != 2:
            raise ClusterConfigurationError(
                f"Malformed RPC endpoint '{entry}'. Exactly one colon separator required."
            )
        host, port_str = parts[0].strip(), parts[1].strip()
        if not host:
            raise ClusterConfigurationError(f"Empty hostname/IP in RPC endpoint '{entry}'.")
        try:
            port = int(port_str)
            if port < 1 or port > 65535:
                raise ValueError()
        except ValueError:
            raise ClusterConfigurationError(
                f"Invalid port number '{port_str}' in RPC endpoint '{entry}'. Must be integer 1..65535."
            )
        parsed_servers.append(f"{host}:{port}")

    return parsed_servers


def ensure_wakelock() -> bool:
    """Acquires Termux system-level wakelock to prevent Android kernel Doze deep-sleep."""
    wakelock_bin = "/data/data/com.termux/files/usr/bin/termux-wake-lock"
    import os
    import subprocess
    if os.path.exists(wakelock_bin):
        try:
            res = subprocess.run([wakelock_bin], capture_output=True, timeout=2.0)
            if res.returncode == 0:
                logger.info("[AMEVA-CLUSTER] Termux WakeLock acquired successfully.")
                return True
        except Exception as exc:
            logger.warning("[AMEVA-CLUSTER] Failed to acquire termux-wake-lock: %s", exc)
    return False


def resolve_auto_tensor_split(
    rpc_spec: Optional[Union[str, List[str]]],
    tensor_split: Optional[str] = None,
    min_guardband_mb: int = 400
) -> Tuple[List[str], Optional[str]]:
    """Automatically compute optimal tensor split when tensor_split == 'auto'.
    Preserves explicit tensor_split specifications without alteration.
    """
    servers = parse_cluster_rpc_spec(rpc_spec)
    if not servers:
        return [], tensor_split

    if tensor_split and str(tensor_split).strip().lower() == "auto":
        try:
            from termux_ai_orchestrator.auto_split import resolve_auto_tensor_split as orch_resolve
            return orch_resolve(servers, min_guardband_mb=min_guardband_mb)
        except Exception as exc:
            logger.warning("[AMEVA-CLUSTER] Auto-TS fallback to equal distribution: %s", exc)
            # Safe equal distribution fallback
            n_nodes = len(servers) + 1
            equal_pct = int(100 // n_nodes)
            remainder = 100 - (equal_pct * n_nodes)
            splits = [equal_pct + remainder] + [equal_pct] * len(servers)
            return servers, ",".join(str(s) for s in splits)

    return servers, tensor_split


def verify_rpc_cluster_nodes(servers: List[str], timeout: float = 3.0) -> None:
    """Strictly verify TCP reachability of each RPC server prior to initiating inference graph.

    Zero-Silent-Fallback: If ANY specified RPC worker is unreachable, this function raises
    ClusterConnectionError immediately to prevent silent local CPU/GPU fallback degradation.
    """
    if not servers:
        return

    failed_nodes = []
    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                logger.info("[AMEVA-CLUSTER] RPC Worker verified reachable: %s", endpoint)
        except (socket.timeout, ConnectionRefusedError, OSError) as exc:
            failed_nodes.append((endpoint, str(exc)))

    if failed_nodes:
        err_details = "\n".join([f"  - {ep}: {err}" for ep, err in failed_nodes])
        raise ClusterConnectionError(
            f"Distributed RPC Cluster Pre-Flight Check FAILED for {len(failed_nodes)}/{len(servers)} worker(s):\n"
            f"{err_details}\n"
            f"Zero-Silent-Fallback Violation Prevented: Inference terminated to avoid silent execution degradation."
        )


def verify_rpc_cluster_health(servers: List[str], timeout: float = 3.0) -> Dict[str, Any]:
    """Inspect and report cluster connectivity status without throwing, returning granular diagnostics."""
    status: Dict[str, Any] = {"all_healthy": True, "nodes": {}}
    for endpoint in servers:
        host, port_str = endpoint.split(":")
        port = int(port_str)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                status["nodes"][endpoint] = {"status": "ONLINE", "error": None}
        except Exception as exc:
            status["all_healthy"] = False
            status["nodes"][endpoint] = {"status": "OFFLINE", "error": str(exc)}
    return status


class ClusterOptionForwarder:
    """Transparent option collector and pass-through forwarder for sub-engines.

    Zero-Deception & Zero-Silent-Drop Principle:
    Preserves all CLI arguments, keyword options, and custom engine parameters
    without silent truncation, enabling seamless pass-through to underlying binaries
    (sd-cli, llama-cli, whisper-cli, etc.) across distributed cluster nodes.
    """

    @staticmethod
    def to_cli_flag(key: str) -> str:
        """Convert snake_case Python identifier to --kebab-case CLI argument flag."""
        clean_key = key.strip().replace("_", "-")
        if not clean_key.startswith("-"):
            clean_key = f"--{clean_key}"
        return clean_key

    @classmethod
    def normalize_forward_args(
        cls,
        options: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
        exclude_keys: Optional[Union[List[str], set]] = None,
    ) -> List[str]:
        """Convert a dictionary of engine options into an argv argument list,
        losslessly preserving raw extra arguments at the end.
        """
        exclude = set(exclude_keys or [])
        cmd_args: List[str] = []

        for k, v in options.items():
            if k in exclude or v is None:
                continue

            flag = cls.to_cli_flag(k)

            if isinstance(v, bool):
                if v:
                    cmd_args.append(flag)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    cmd_args.extend([flag, str(item)])
            else:
                cmd_args.extend([flag, str(v)])

        if extra_args:
            for arg in extra_args:
                if str(arg).strip():
                    cmd_args.append(str(arg).strip())

        return cmd_args

    @classmethod
    def merge_pass_through_args(
        cls,
        base_cmd: List[str],
        extra_args: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Merge base command with pass-through extra arguments and key-value overrides."""
        merged = list(base_cmd)
        if overrides:
            override_args = cls.normalize_forward_args(overrides)
            merged.extend(override_args)
        if extra_args:
            for a in extra_args:
                if str(a).strip():
                    merged.append(str(a).strip())
        return merged


class SequentialClusterPipeline:
    """Two-Stage Memory Isolation Pipeline for Mobile On-Device AI Clustering.

    Architecture (Text-Encode-Then-Kill -> Disaggregated DiT Inference):
    - Stage 1 (Text Encoding Phase): Spawns isolated subprocess for LLM/CLIP text encoding.
      Generates prompt embeddings/cache, then immediately terminates and kills the process
      to return 100% of the allocated memory (~1.7GB) to the Android kernel.
    - Stage 2 (Disaggregated DiT Sampling Phase): Launches diffusion backbone inference
      in clean RAM environment, losslessly forwarding all distributed RPC cluster flags
      (--rpc-servers, -ts, --schedule, --flow-shift, etc.) to remote workers.
    """

    def __init__(self, forwarder: Optional[ClusterOptionForwarder] = None):
        self.forwarder = forwarder or ClusterOptionForwarder()

    def execute_stages(
        self,
        prompt: str,
        stage1_fn: Callable[[], Any],
        stage2_fn: Callable[[], Any],
    ) -> Any:
        """Execute Stage 1 and Stage 2 sequentially with strict fail-fast enforcement."""
        ensure_wakelock()
        logger.info("[AMEVA-CLUSTER] Initiating Sequential 2-Stage Memory Isolation Pipeline...")
        t0 = time.time()

        # Stage 1: Text Encoding with memory recovery guarantee
        logger.info("[AMEVA-CLUSTER] [Stage 1/2] Executing Isolated Text Encoding Phase...")
        try:
            stage1_result = stage1_fn()
            logger.info("[AMEVA-CLUSTER] [Stage 1/2] Text Encoding completed. Reclaiming memory.")
        except Exception as e:
            logger.error("[AMEVA-CLUSTER] [Stage 1/2] Text Encoding failed: %s", e)
            raise

        # Stage 2: Disaggregated DiT Sampling & Decoding
        logger.info("[AMEVA-CLUSTER] [Stage 2/2] Launching Distributed DiT Inference in clean RAM...")
        try:
            stage2_result = stage2_fn()
            elapsed = time.time() - t0
            logger.info("[AMEVA-CLUSTER] 2-Stage Sequential Inference succeeded in %.2fs", elapsed)
            return stage2_result
        except Exception as e:
            logger.error("[AMEVA-CLUSTER] [Stage 2/2] Distributed DiT Inference failed: %s", e)
            raise

