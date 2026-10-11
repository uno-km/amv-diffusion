# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""ComfyUI Workflow (DAG) Dual-Format Engine for termux-diffusion.

Parses, topologically sorts, and executes ComfyUI workflows in both:
1. UI Graph Format (standard exported JSON with 'nodes' and 'links')
2. API Prompt Format (clean dictionary keyed by node ID with 'class_type' and 'inputs')

Follows the 100% Open & Scalable Architecture Principle:
Zero artificial hardware blocking. Hardware scalability scales from 6GB up to 24GB/32GB+
devices with dynamic resource adaptation and extensible node dispatchers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("termux_diffusion.workflow")


class WorkflowError(Exception):
    """Base exception for workflow parsing and execution failures."""
    pass


class WorkflowFormatError(WorkflowError):
    """Raised when workflow JSON structure is invalid or unparseable."""
    pass


class WorkflowCycleError(WorkflowError):
    """Raised when cyclic dependency is detected in the DAG."""
    pass


class UnsupportedNodeError(WorkflowError):
    """Raised when an unhandled node is encountered and cannot be bypassed."""
    pass


@dataclass
class NodeConnection:
    """Represents a connection from source node slot to target node input."""
    source_node_id: str
    source_slot_index: int = 0


@dataclass
class WorkflowNode:
    """Standardized representation of an executable node in the DAG."""
    id: str
    class_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    widgets_values: List[Any] = field(default_factory=list)
    title: Optional[str] = None
    output_types: List[str] = field(default_factory=list)


@dataclass
class WorkflowExecutionResult:
    """Encapsulates the final outcome and artifacts of a workflow run."""
    success: bool
    output_images: List[Path]
    executed_nodes: List[str]
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeRegistry:
    """Extensible registry for node executors with pluggable fallback."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[WorkflowNode, Dict[str, Any], "ExecutionContext"], Any]] = {}

    def register(self, *class_names: str) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            for name in class_names:
                self._handlers[name.lower()] = fn
            return fn
        return decorator

    def get(self, class_name: str) -> Optional[Callable]:
        return self._handlers.get(class_name.lower())

    def list_supported_nodes(self) -> List[str]:
        return sorted(list(self._handlers.keys()))


# Global node registry instance
NODE_REGISTRY = NodeRegistry()


@dataclass
class ExecutionContext:
    """Runtime state passed across topologically sorted node executions."""
    workflow_path: Optional[Path]
    output_dir: Path
    device: str
    force_run: bool
    progress: bool
    json_progress: bool
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    generated_images: List[Path] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class WorkflowParser:
    """Detects and parses ComfyUI UI Graph Format and API Prompt Format."""

    @staticmethod
    def load_json(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Loads JSON from file path or validates raw dictionary."""
        if isinstance(source, (str, Path)):
            p = Path(source).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"[WorkflowParser] Workflow JSON file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as err:
                    raise WorkflowFormatError(f"Corrupt JSON syntax in '{p}': {err}") from err
        elif isinstance(source, dict):
            return source
        else:
            raise WorkflowFormatError(f"Expected file path or dictionary, got {type(source)}")

    @staticmethod
    def detect_format(data: Dict[str, Any]) -> str:
        """Determines whether payload is 'UI_GRAPH' or 'API_PROMPT'."""
        if "nodes" in data and isinstance(data["nodes"], list) and "links" in data:
            return "UI_GRAPH"
        # Check API format: dictionary where keys map to objects with class_type & inputs
        if data and all(isinstance(v, dict) and "class_type" in v and "inputs" in v for v in data.values()):
            return "API_PROMPT"
        # Fallback check for nested prompt format
        if "prompt" in data and isinstance(data["prompt"], dict):
            return "API_PROMPT"
        raise WorkflowFormatError(
            "[WorkflowParser] Unknown workflow format. File must either contain 'nodes' & 'links' (UI Format) "
            "or node dictionaries with 'class_type' and 'inputs' (API Prompt Format)."
        )

    @classmethod
    def parse(cls, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, WorkflowNode]:
        """Parses arbitrary ComfyUI workflow JSON into canonical WorkflowNode graph."""
        data = cls.load_json(source)
        fmt = cls.detect_format(data)
        logger.debug("[WorkflowParser] Detected workflow format: %s", fmt)

        if fmt == "UI_GRAPH":
            inlined_data = cls.inline_subgraphs(data)
            return cls._parse_ui_graph(inlined_data)
        elif fmt == "API_PROMPT":
            prompt_data = data.get("prompt", data)
            return cls._parse_api_prompt(prompt_data)
        raise WorkflowFormatError(f"Unsupported format: {fmt}")

    @staticmethod
    def inline_subgraphs(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inlines ComfyUI v0.4+ definitions.subgraphs into root nodes and links.

        Zero-Silent-Fallback Principle: Wrapper node slot connections, widget parameters,
        and subgraph-internal linkages are mechanically preserved and rerouted without loss.
        """
        import copy
        data = copy.deepcopy(raw_data)
        definitions = data.get("definitions", {})
        subgraphs_list = definitions.get("subgraphs", [])
        if not subgraphs_list or not isinstance(subgraphs_list, list):
            return data

        subgraphs_by_id = {sg["id"]: sg for sg in subgraphs_list if isinstance(sg, dict) and "id" in sg}
        if not subgraphs_by_id:
            return data

        nodes = data.get("nodes", [])
        links = data.get("links", [])

        link_map: Dict[int, List[Any]] = {}
        max_link_id = 0
        for l in links:
            if isinstance(l, (list, tuple)) and len(l) >= 5:
                link_map[l[0]] = list(l)
                if isinstance(l[0], int) and l[0] > max_link_id:
                    max_link_id = l[0]

        new_nodes: List[Dict[str, Any]] = []
        new_links = list(links)

        for n in nodes:
            if not isinstance(n, dict):
                continue
            ntype = str(n.get("type", n.get("class_type", "")))
            if ntype in subgraphs_by_id:
                sg = subgraphs_by_id[ntype]
                wrapper_id = n["id"]
                wrapper_inputs = n.get("inputs", [])
                wrapper_wvals_named = n.get("widgets_values_named", {})
                wrapper_wvals = n.get("widgets_values", [])

                # Map input slot index -> incoming link
                input_link_by_slot: Dict[int, int] = {}
                for idx, inp in enumerate(wrapper_inputs):
                    if inp.get("link") is not None:
                        input_link_by_slot[idx] = inp["link"]

                sg_inputs = sg.get("inputs", [])
                sg_outputs = sg.get("outputs", [])
                sg_links = sg.get("links", [])
                sg_nodes = copy.deepcopy(sg.get("nodes", []))

                # Internal links
                internal_links_from_input: Dict[int, List[Tuple[Any, int, str, int]]] = {}
                internal_links_to_output: Dict[int, Tuple[Any, int, str, int]] = {}

                for sl in sg_links:
                    lid = sl.get("id")
                    oid = sl.get("origin_id")
                    oslot = sl.get("origin_slot")
                    tid = sl.get("target_id")
                    tslot = sl.get("target_slot")
                    ltype = sl.get("type", "ANY")

                    if oid == -10:
                        internal_links_from_input.setdefault(oslot, []).append((tid, tslot, ltype, lid))
                    elif tid == -20:
                        internal_links_to_output[tslot] = (oid, oslot, ltype, lid)
                    else:
                        new_links.append([lid, oid, oslot, tid, tslot, ltype])

                # Rewire wrapper inputs
                for s_idx, sg_inp in enumerate(sg_inputs):
                    inp_name = sg_inp.get("name")
                    outer_link_id = input_link_by_slot.get(s_idx)
                    targets = internal_links_from_input.get(s_idx, [])

                    if outer_link_id is not None and outer_link_id in link_map:
                        ol = link_map[outer_link_id]
                        for t_i, (tid, tslot, ltype, orig_lid) in enumerate(targets):
                            if t_i == 0:
                                ol[3] = tid
                                ol[4] = tslot
                                for sn in sg_nodes:
                                    if sn["id"] == tid:
                                        if "inputs" in sn and len(sn["inputs"]) > tslot:
                                            sn["inputs"][tslot]["link"] = outer_link_id
                            else:
                                max_link_id += 1
                                new_links.append([max_link_id, ol[1], ol[2], tid, tslot, ltype])
                                for sn in sg_nodes:
                                    if sn["id"] == tid:
                                        if "inputs" in sn and len(sn["inputs"]) > tslot:
                                            sn["inputs"][tslot]["link"] = max_link_id
                    else:
                        # Extract literal value from wrapper parameters
                        val = None
                        if isinstance(wrapper_wvals_named, dict) and inp_name in wrapper_wvals_named:
                            val = wrapper_wvals_named[inp_name]
                        elif s_idx < len(wrapper_wvals):
                            val = wrapper_wvals[s_idx]

                        if val is not None:
                            for tid, tslot, ltype, orig_lid in targets:
                                for sn in sg_nodes:
                                    if sn["id"] == tid:
                                        if "inputs" in sn and len(sn["inputs"]) > tslot:
                                            sn["inputs"][tslot]["link"] = None
                                            sn["inputs"][tslot]["default_value"] = val
                                        sn.setdefault("widgets_values_named", {})[inp_name] = val
                                        w_v = sn.setdefault("widgets_values", [])
                                        if tslot < len(w_v):
                                            w_v[tslot] = val
                                        else:
                                            w_v.append(val)

                # Rewire wrapper outputs
                for out_slot_idx, sg_out in enumerate(sg_outputs):
                    if out_slot_idx in internal_links_to_output:
                        oid, oslot, ltype, _ = internal_links_to_output[out_slot_idx]
                        for l in new_links:
                            if l[1] == wrapper_id and l[2] == out_slot_idx:
                                l[1] = oid
                                l[2] = oslot

                new_nodes.extend(sg_nodes)
            else:
                new_nodes.append(n)

        # Filter out links referencing replaced wrapper node
        final_links = [l for l in new_links if l[1] not in subgraphs_by_id and l[3] not in subgraphs_by_id]
        data["nodes"] = new_nodes
        data["links"] = final_links
        return data

    @staticmethod
    def _parse_ui_graph(data: Dict[str, Any]) -> Dict[str, WorkflowNode]:
        """Parses ComfyUI UI format using 'nodes' and 'links' table."""
        nodes_raw = data.get("nodes", [])
        links_raw = data.get("links", [])

        # Link index: link_id -> (from_node_id, from_slot, to_node_id, to_slot, type_name)
        link_map: Dict[int, Tuple[str, int, str, int, str]] = {}
        for link in links_raw:
            if isinstance(link, (list, tuple)) and len(link) >= 5:
                l_id = link[0]
                from_node = str(link[1])
                from_slot = int(link[2])
                to_node = str(link[3])
                to_slot = int(link[4])
                t_name = str(link[5]) if len(link) > 5 else "ANY"
                link_map[l_id] = (from_node, from_slot, to_node, to_slot, t_name)

        nodes: Dict[str, WorkflowNode] = {}
        for n in nodes_raw:
            if not isinstance(n, dict) or "id" not in n:
                continue
            node_id = str(n["id"])
            class_type = str(n.get("type", n.get("class_type", "UnknownNode")))
            widgets_vals = list(n.get("widgets_values") or [])
            title = n.get("title")

            inputs: Dict[str, Any] = {}
            # Wire up connected input slots or default inlined values
            for inp_slot in n.get("inputs", []):
                inp_name = inp_slot.get("name")
                link_id = inp_slot.get("link")
                if link_id is not None and link_id in link_map:
                    from_node, from_slot, _, _, _ = link_map[link_id]
                    inputs[inp_name] = NodeConnection(source_node_id=from_node, source_slot_index=from_slot)
                elif "default_value" in inp_slot:
                    inputs[inp_name] = inp_slot["default_value"]

            out_types = []
            for out_slot in n.get("outputs", []):
                out_types.append(out_slot.get("type", "ANY"))

            nodes[node_id] = WorkflowNode(
                id=node_id,
                class_type=class_type,
                inputs=inputs,
                widgets_values=widgets_vals,
                title=title,
                output_types=out_types
            )
        return nodes

    @staticmethod
    def _parse_api_prompt(data: Dict[str, Any]) -> Dict[str, WorkflowNode]:
        """Parses ComfyUI API Prompt format."""
        nodes: Dict[str, WorkflowNode] = {}
        for node_id_raw, node_dict in data.items():
            if not isinstance(node_dict, dict):
                continue
            node_id = str(node_id_raw)
            class_type = str(node_dict.get("class_type", "UnknownNode"))
            raw_inputs = node_dict.get("inputs", {})

            inputs: Dict[str, Any] = {}
            for k, v in raw_inputs.items():
                if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int):
                    inputs[k] = NodeConnection(source_node_id=str(v[0]), source_slot_index=int(v[1]))
                else:
                    inputs[k] = v

            nodes[node_id] = WorkflowNode(
                id=node_id,
                class_type=class_type,
                inputs=inputs,
                widgets_values=[],
                title=node_dict.get("_meta", {}).get("title") if isinstance(node_dict.get("_meta"), dict) else None
            )
        return nodes


class DAGEngine:
    """Manages DAG dependency resolution, cycle detection, and topological sorting."""

    @staticmethod
    def topological_sort(nodes: Dict[str, WorkflowNode]) -> List[str]:
        """Performs Kahn's algorithm to resolve node execution order."""
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {node_id: 0 for node_id in nodes}

        for node_id, node in nodes.items():
            for inp_val in node.inputs.values():
                if isinstance(inp_val, NodeConnection):
                    src_id = inp_val.source_node_id
                    if src_id in nodes:
                        adj[src_id].append(node_id)
                        in_degree[node_id] += 1

        queue: deque[str] = deque([nid for nid, deg in in_degree.items() if deg == 0])
        ordered: List[str] = []

        while queue:
            curr = queue.popleft()
            ordered.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(nodes):
            unresolved = [nid for nid, deg in in_degree.items() if deg > 0]
            raise WorkflowCycleError(
                f"[DAGEngine] Cyclic dependency or unreachable graph cycle detected in nodes: {unresolved}"
            )
        return ordered


# --- Standard ComfyUI Node Handlers Implementation ---

@NODE_REGISTRY.register(
    "checkpointloadersimple",
    "checkpointloader",
    "dualcliploader",
    "cliploader",
    "unetloader",
    "vaeloader",
    "loadcheckpoint"
)
def handle_checkpoint_loader(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Resolves and loads model weights (GGUF or Safetensors) with Zero-Silent-Fallback enforcement."""
    ckpt_name = None
    for key in ("ckpt_name", "unet_name", "clip_name", "vae_name", "model_name"):
        if key in inputs:
            ckpt_name = str(inputs[key])
            break
    if not ckpt_name and node.widgets_values:
        ckpt_name = str(node.widgets_values[0])

    if not ckpt_name:
        ckpt_name = "realistic"

    # Enforce Zero-Silent-Fallback: If an uncataloged specific weight file (.safetensors, .gguf, or path) is requested,
    # strictly verify its presence on disk. Silent weight substitution or mock fabrication is strictly prohibited.
    if isinstance(ckpt_name, str) and (ckpt_name.endswith(".safetensors") or ckpt_name.endswith(".gguf") or "/" in ckpt_name):
        from .hub import get_cache_dir, is_model_cached, list_presets
        presets = list_presets()
        is_known_preset = ckpt_name in presets or any(v.get("alias") == ckpt_name for v in presets.values() if isinstance(v, dict))

        if not is_known_preset:
            cache_dir = get_cache_dir()
            possible_paths = [
                Path(ckpt_name),
                Path.cwd() / ckpt_name,
                Path.cwd() / "models" / ckpt_name,
                Path.cwd() / "models" / "diffusion_models" / ckpt_name,
                Path.cwd() / "models" / "text_encoders" / ckpt_name,
                Path.cwd() / "models" / "vae" / ckpt_name,
                cache_dir / ckpt_name,
                cache_dir / "models" / ckpt_name,
            ]
            found = any(p.is_file() for p in possible_paths) or is_model_cached(ckpt_name)
            if not found and not ctx.force_run:
                raise FileNotFoundError(
                    f"[Workflow] [Node {node.id}: {node.class_type}] Required weight file '{ckpt_name}' was not found on device. "
                    f"Zero-Silent-Fallback Protocol: Silent weight remapping or mock substitution is strictly prohibited. "
                    f"Please ensure '{ckpt_name}' is placed in models/ or downloaded before execution."
                )

    logger.info("[Workflow] [Node %s: %s] Checkpoint resolved: %s", node.id, node.class_type, ckpt_name)
    return {
        "model": ckpt_name,
        "clip": ckpt_name,
        "vae": ckpt_name
    }


@NODE_REGISTRY.register("cliptextencode", "cliptextencodesdxl")
def handle_clip_text_encode(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Extracts text prompt conditioning."""
    text = ""
    if "text" in inputs:
        text = str(inputs["text"])
    elif node.widgets_values:
        text = str(node.widgets_values[0])

    logger.info("[Workflow] [Node %s: %s] Prompt text: '%s'", node.id, node.class_type, text[:40] + ("..." if len(text) > 40 else ""))
    return {"conditioning": text}


@NODE_REGISTRY.register("conditioningzeroout")
def handle_conditioning_zero_out(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Zeroes out text conditioning vector (sets negative prompt to empty string for CFG=1 distillation)."""
    logger.info("[Workflow] [Node %s: %s] Conditioning zeroed out for CFG=1 guidance distillation", node.id, node.class_type)
    return {"conditioning": ""}


@NODE_REGISTRY.register("emptylatentimage", "emptysd3latentimage")
def handle_empty_latent(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Sets initial canvas dimensions."""
    width = 512
    height = 512
    batch_size = 1

    if "width" in inputs:
        w_val = inputs["width"]
        if isinstance(w_val, dict) and "width" in w_val:
            width = int(w_val["width"])
        else:
            try:
                width = int(w_val)
            except (ValueError, TypeError):
                pass
    elif len(node.widgets_values) >= 1:
        try:
            width = int(node.widgets_values[0])
        except (ValueError, TypeError):
            pass

    if "height" in inputs:
        h_val = inputs["height"]
        if isinstance(h_val, dict) and "height" in h_val:
            height = int(h_val["height"])
        else:
            try:
                height = int(h_val)
            except (ValueError, TypeError):
                pass
    elif len(node.widgets_values) >= 2:
        try:
            height = int(node.widgets_values[1])
        except (ValueError, TypeError):
            pass

    if "batch_size" in inputs:
        try:
            batch_size = int(inputs["batch_size"])
        except (ValueError, TypeError):
            pass
    elif len(node.widgets_values) >= 3:
        try:
            batch_size = int(node.widgets_values[2])
        except (ValueError, TypeError):
            pass

    logger.info("[Workflow] [Node %s: %s] Canvas: %dx%d (Batch: %d)", node.id, node.class_type, width, height, batch_size)
    return {"latent": {"width": width, "height": height, "batch_size": batch_size}}


@NODE_REGISTRY.register("modelsamplingauraflow", "modelsamplingflux")
def handle_model_sampling(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Configures model sampling schedule parameters (e.g. AuraFlow/Flux shift)."""
    model_in = inputs.get("model", {})
    shift = 3.0
    if "shift" in inputs:
        try:
            shift = float(inputs["shift"])
        except (ValueError, TypeError):
            pass
    elif node.widgets_values:
        try:
            shift = float(node.widgets_values[0])
        except (ValueError, IndexError, TypeError):
            pass

    res = dict(model_in) if isinstance(model_in, dict) else {"model": str(model_in)}
    res["shift"] = shift
    logger.info("[Workflow] [Node %s: %s] ModelSampling shift configured: %s", node.id, node.class_type, shift)
    return {"model": res}


@NODE_REGISTRY.register("resolutionselector")
def handle_resolution_selector(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Resolves aspect ratio string to width and height dimensions."""
    ratio_str = "1:1 (Square)"
    if "aspect_ratio" in inputs:
        ratio_str = str(inputs["aspect_ratio"])
    elif node.widgets_values:
        ratio_str = str(node.widgets_values[0])

    ASPECT_RATIOS = {
        "1:1": (1024, 1024),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
        "21:9": (1536, 640),
    }

    width, height = 1024, 1024
    for key, (w, h) in ASPECT_RATIOS.items():
        if key in ratio_str:
            width, height = w, h
            break

    logger.info("[Workflow] [Node %s: %s] Selected resolution: %dx%d for '%s'", node.id, node.class_type, width, height, ratio_str)
    return {
        "width": width,
        "height": height,
        0: width,
        1: height
    }


@NODE_REGISTRY.register("markdownnote", "note")
def handle_note(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Informational documentation note node (no-op)."""
    return {}


@NODE_REGISTRY.register("ksampler", "ksampleradvanced", "samplercustomadvanced")
def handle_ksampler(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Coordinates and executes the primary diffusion inference step."""
    from .core import generate

    # Extract model and shift
    model_obj = inputs.get("model", {})
    shift_val = None
    while isinstance(model_obj, dict):
        if "shift" in model_obj and shift_val is None:
            shift_val = model_obj["shift"]
        if "model" in model_obj and isinstance(model_obj["model"], (dict, str)):
            model_obj = model_obj["model"]
        else:
            break
    model_name = str(model_obj) if model_obj and not isinstance(model_obj, dict) else (str(model_obj.get("model", "realistic")) if isinstance(model_obj, dict) else "realistic")
    flow_shift = shift_val if shift_val is not None else (model_obj.get("shift") if isinstance(model_obj, dict) else None)

    # Extract positive/negative conditioning
    pos_obj = inputs.get("positive", {})
    pos_text = pos_obj.get("conditioning", "") if isinstance(pos_obj, dict) else str(pos_obj)

    neg_obj = inputs.get("negative", {})
    neg_text = neg_obj.get("conditioning", "") if isinstance(neg_obj, dict) else str(neg_obj)

    # Extract canvas parameters
    latent_obj = inputs.get("latent_image", {})
    latent_data = latent_obj.get("latent", {}) if isinstance(latent_obj, dict) else {}
    width = latent_data.get("width", 512)
    height = latent_data.get("height", 512)

    # Sampler hyperparameters
    seed = inputs.get("seed", -1)
    steps = inputs.get("steps", 15)
    cfg = inputs.get("cfg", 7.0)
    sampler_name = inputs.get("sampler_name", "euler_a")
    scheduler = inputs.get("scheduler", "default" if "z_image" in str(model_name).lower() else "karras")
    denoise = inputs.get("denoise", 1.0)

    # If parsing from UI widgets values
    if node.widgets_values and len(node.widgets_values) >= 6:
        try:
            seed = int(node.widgets_values[0])
            steps = int(node.widgets_values[2])
            cfg = float(node.widgets_values[3])
            sampler_name = str(node.widgets_values[4])
            scheduler = str(node.widgets_values[5])
            if len(node.widgets_values) >= 7:
                denoise = float(node.widgets_values[6])
        except (ValueError, IndexError):
            pass

    out_file = ctx.output_dir / f"workflow_out_{int(time.time())}_{node.id}.png"

    logger.info(
        "[Workflow] [Node %s: %s] Executing KSampler (steps=%s, cfg=%s, sampler=%s, sched=%s)",
        node.id, node.class_type, steps, cfg, sampler_name, scheduler
    )

    if flow_shift is None and isinstance(model_obj, dict):
        flow_shift = model_obj.get("shift")

    # Mobile memory defense parameters for high-footprint DiT models (Z-Image Turbo 6B)
    extra_kws = dict(ctx.extra_params or {})
    model_str = str(model_name).lower()
    if "z_image" in model_str or "turbo" in model_str or "dit" in model_str:
        extra_kws.setdefault("params_backend", "diffusion=cpu")
        extra_kws.setdefault("max_vram", "vulkan0=1")
        extra_kws.setdefault("stream_layers", True)
        extra_kws.setdefault("clip_on_cpu", True)
        extra_kws.setdefault("vae_on_cpu", True)
        extra_kws.setdefault("vae_format", "flux")
        extra_kws.setdefault("mmap", True)
        extra_kws.setdefault("taesd", "taef1.safetensors")
        extra_kws.setdefault("vae_tiling", True)
        extra_kws.setdefault("safe_mode", True)
        extra_kws.setdefault("gpu_duty_cycle", 90)
        extra_kws.setdefault("diffusion_fa", True)
        extra_kws.setdefault("threads", 4)
        # Use ultra-quantized IQ1_S (1.1GB vs 1.6GB Q2_K) to reclaim 500MB headroom on 8GB phones
        extra_kws.setdefault("llm", "Qwen3-4B-Instruct-2507-Q2_K.gguf")

    gen_res = generate(
        prompt=pos_text or "beautiful realistic masterpiece",
        negative_prompt=neg_text,
        model=model_name,
        device=ctx.device,
        steps=steps,
        cfg_scale=cfg,
        width=width,
        height=height,
        seed=seed,
        sampling_method=sampler_name,
        schedule=scheduler,
        flow_shift=flow_shift,
        output=str(out_file),
        timeout=max(7200, steps * 900),
        progress=ctx.progress,
        json_progress=ctx.json_progress,
        **extra_kws
    )

    ctx.generated_images.append(gen_res.path)
    return {
        "latent": {"path": gen_res.path, "width": width, "height": height},
        "image": gen_res.path
    }


@NODE_REGISTRY.register("vaedecode", "vaedecodetiled")
def handle_vae_decode(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Decodes latent representation to pixel canvas (passthrough if already decoded by KSampler)."""
    samples = inputs.get("samples", {})
    img_path = samples.get("image") or samples.get("path")
    if not img_path and isinstance(samples, dict) and "latent" in samples:
        img_path = samples["latent"].get("path")

    logger.info("[Workflow] [Node %s: %s] VAE Decode pass complete: %s", node.id, node.class_type, img_path)
    return {"image": img_path}


@NODE_REGISTRY.register("facedetailer", "facedetailerpipe", "detailerforeach")
def handle_face_detailer(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Executes on-device FaceDetailer localized crop & Img2Img loop."""
    from .core import generate
    from .detailer import refine_face_regions

    img_input = inputs.get("image")
    if isinstance(img_input, dict):
        img_input = img_input.get("image") or img_input.get("path")

    if not img_input or not Path(str(img_input)).is_file():
        # Fallback to most recent generated image
        if ctx.generated_images:
            img_input = ctx.generated_images[-1]
        else:
            raise WorkflowError(f"[FaceDetailer Node {node.id}] No source image found to refine.")

    steps = int(inputs.get("steps", 2))
    denoise = float(inputs.get("denoise", 0.35))
    guide_size = int(inputs.get("guide_size", 512))

    if node.widgets_values and len(node.widgets_values) >= 3:
        try:
            guide_size = int(node.widgets_values[0])
            steps = int(node.widgets_values[3])
            denoise = float(node.widgets_values[5])
        except (ValueError, IndexError):
            pass

    detailed_out = ctx.output_dir / f"detailed_{Path(img_input).name}"
    logger.info("[Workflow] [Node %s: %s] Applying FaceDetailer on '%s'", node.id, node.class_type, img_input)

    res_path = refine_face_regions(
        image_path=img_input,
        generator_func=generate,
        prompt="beautiful realistic eyes, detailed facial features, flawless natural skin, 8k portrait",
        output_path=detailed_out,
        strength=denoise,
        steps=steps,
        device=ctx.device,
        generator_kwargs={"progress": ctx.progress}
    )

    ctx.generated_images.append(res_path)
    return {"image": res_path}


@NODE_REGISTRY.register("saveimage", "previewimage", "saveimageadvanced")
def handle_save_image(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Exports and finalizes destination images."""
    img_input = inputs.get("images") or inputs.get("image")
    if isinstance(img_input, dict):
        img_input = img_input.get("image") or img_input.get("path")

    target_path = Path(str(img_input)) if img_input else None
    if not target_path or not target_path.is_file():
        if ctx.generated_images:
            target_path = ctx.generated_images[-1]

    logger.info("[Workflow] [Node %s: %s] Final artifact saved: %s", node.id, node.class_type, target_path)
    return {"saved": str(target_path)}


# --- Workflow Runner Interface ---

class WorkflowExecutor:
    """Executes parsed workflow graphs with dynamic resource adaptability."""

    def __init__(self, registry: Optional[NodeRegistry] = None) -> None:
        self.registry = registry or NODE_REGISTRY

    def run(
        self,
        workflow: Union[str, Path, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "auto",
        force_run: bool = False,
        progress: bool = True,
        json_progress: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
        node_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> WorkflowExecutionResult:
        """Executes a ComfyUI workflow end-to-end.

        Args:
            workflow: File path to .json or parsed dictionary.
            output_dir: Target folder for generated outputs.
            device: 'auto', 'gpu', 'vulkan', 'cpu'.
            force_run: Bypass hardware tiering checks (for 16GB/24GB+ workstations).
            progress: Display progress bars.
            json_progress: Stream NDJSON progress events.
            extra_params: Key-value parameter overrides.
            node_progress_callback: Real-time callback reporting active node stage.
        """
        start_time = time.time()
        wf_path = Path(workflow).resolve() if isinstance(workflow, (str, Path)) and os.path.exists(str(workflow)) else None

        # 1. Parse and build DAG
        nodes = WorkflowParser.parse(workflow)
        logger.info("[WorkflowExecutor] Successfully parsed %d nodes from workflow.", len(nodes))

        # 2. Topological sort
        exec_order = DAGEngine.topological_sort(nodes)
        logger.info("[WorkflowExecutor] Execution order resolved (%d steps): %s", len(exec_order), " -> ".join(exec_order))

        # 3. Setup Context
        resolved_out_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "outputs"
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        ctx = ExecutionContext(
            workflow_path=wf_path,
            output_dir=resolved_out_dir,
            device=device,
            force_run=force_run,
            progress=progress,
            json_progress=json_progress,
            extra_params=dict(extra_params or {})
        )

        executed_node_ids: List[str] = []
        total_steps_in_workflow = len(exec_order)

        # 4. Sequential execution loop
        for idx, node_id in enumerate(exec_order, start=1):
            node = nodes[node_id]
            handler = self.registry.get(node.class_type)

            if handler is None:
                if not force_run:
                    raise UnsupportedNodeError(
                        f"[WorkflowExecutor] [UNSUPPORTED_NODE] Node #{node.id} ('{node.class_type}') has no "
                        f"registered handler. Use --force-run to attempt best-effort bypass or register a custom handler."
                    )
                logger.warning(
                    "[WorkflowExecutor] Bypassing unregistered node #%s ('%s') under --force-run.",
                    node.id, node.class_type
                )
                continue

            # Resolve input values from predecessor outputs
            resolved_inputs: Dict[str, Any] = {}
            for in_name, in_val in node.inputs.items():
                if isinstance(in_val, NodeConnection):
                    src_out = ctx.node_outputs.get(in_val.source_node_id, {})
                    if isinstance(src_out, dict):
                        resolved_inputs[in_name] = src_out
                    else:
                        resolved_inputs[in_name] = src_out
                else:
                    resolved_inputs[in_name] = in_val

            # Apply runtime overrides if matching node id or class type
            if ctx.extra_params:
                for k, v in ctx.extra_params.items():
                    if k in resolved_inputs:
                        resolved_inputs[k] = v

            # Notify node execution start
            if node_progress_callback:
                node_progress_callback({
                    "current_node_index": idx,
                    "total_nodes": total_steps_in_workflow,
                    "node_id": node.id,
                    "node_title": node.title or node.class_type,
                    "class_type": node.class_type,
                    "phase": f"Node {idx}/{total_steps_in_workflow}: {node.title or node.class_type}",
                    "percent": round(((idx - 1) / total_steps_in_workflow) * 100, 1),
                    "status": "executing",
                    "executed_nodes": list(executed_node_ids)
                })

            # Execute node
            logger.debug("[WorkflowExecutor] Running node #%s: %s", node.id, node.class_type)
            result = handler(node, resolved_inputs, ctx)
            ctx.node_outputs[node.id] = result
            executed_node_ids.append(node.id)

            # Notify node execution complete
            if node_progress_callback:
                node_progress_callback({
                    "current_node_index": idx,
                    "total_nodes": total_steps_in_workflow,
                    "node_id": node.id,
                    "node_title": node.title or node.class_type,
                    "class_type": node.class_type,
                    "phase": f"Completed Node {idx}/{total_steps_in_workflow}: {node.title or node.class_type}",
                    "percent": round((idx / total_steps_in_workflow) * 100, 1),
                    "status": "completed",
                    "executed_nodes": list(executed_node_ids)
                })

        elapsed = time.time() - start_time
        return WorkflowExecutionResult(
            success=True,
            output_images=ctx.generated_images,
            executed_nodes=executed_node_ids,
            elapsed_seconds=elapsed,
            metadata={"total_nodes": len(nodes), "format": WorkflowParser.detect_format(WorkflowParser.load_json(workflow))}
        )


def execute_workflow(
    workflow_path: Union[str, Path, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    device: str = "auto",
    force_run: bool = False,
    progress: bool = True,
    json_progress: bool = False,
    node_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs: Any
) -> WorkflowExecutionResult:
    """Convenience helper to execute a workflow with default settings."""
    executor = WorkflowExecutor()
    return executor.run(
        workflow=workflow_path,
        output_dir=output_dir,
        device=device,
        force_run=force_run,
        progress=progress,
        json_progress=json_progress,
        extra_params=kwargs,
        node_progress_callback=node_progress_callback
    )


def get_workflows_dir() -> Path:
    """Returns the bundled workflows directory."""
    return Path(__file__).resolve().parent / "workflows"


def get_user_workflows_dir() -> Path:
    """Returns the user custom workflows directory under ~/.cache/termux-diffusion/workflows."""
    d = Path.home() / ".cache" / "termux-diffusion" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_builtin_workflows() -> List[Dict[str, Any]]:
    """Lists standard bundled and user-saved AmfyUI / ComfyUI workflows."""
    results: List[Dict[str, Any]] = []
    seen_ids = set()

    # 1. User-saved custom workflows first (higher priority)
    user_dir = get_user_workflows_dir()
    if user_dir.is_dir():
        for p in sorted(user_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nodes = WorkflowParser.parse(data)
                order = DAGEngine.topological_sort(nodes)
                wf_id = p.stem
                name = data.get("name", wf_id.replace("_", " ").title())
                desc = data.get("description", "User Custom Workflow")
                author = data.get("author", "User")
                seen_ids.add(wf_id)
                results.append({
                    "id": wf_id,
                    "name": name,
                    "description": desc,
                    "author": author,
                    "node_count": len(nodes),
                    "execution_order": order,
                    "file_path": str(p),
                    "custom": not (get_workflows_dir() / f"{wf_id}.json").is_file()
                })
            except Exception as e:
                logger.warning("[list_builtin_workflows] Failed parsing user workflow %s: %s", p, e)

    # 2. Bundled canonical workflows
    wf_dir = get_workflows_dir()
    if wf_dir.is_dir():
        for p in sorted(wf_dir.glob("*.json")):
            wf_id = p.stem
            if wf_id in seen_ids:
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nodes = WorkflowParser.parse(data)
                order = DAGEngine.topological_sort(nodes)
                name = data.get("name", wf_id.replace("_", " ").title())
                desc = data.get("description", "Standard AmfyUI Workflow")
                author = data.get("author", "AmfyUI")
                results.append({
                    "id": wf_id,
                    "name": name,
                    "description": desc,
                    "author": author,
                    "node_count": len(nodes),
                    "execution_order": order,
                    "file_path": str(p),
                    "custom": False
                })
            except Exception as e:
                logger.warning("[list_builtin_workflows] Failed parsing %s: %s", p, e)
    return results


def get_builtin_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Loads and returns workflow JSON dictionary by ID, searching user dir then bundled dir."""
    user_target = get_user_workflows_dir() / f"{workflow_id}.json"
    if user_target.is_file():
        with open(user_target, "r", encoding="utf-8") as f:
            return json.load(f)

    wf_dir = get_workflows_dir()
    target = wf_dir / f"{workflow_id}.json"
    if target.is_file():
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_user_workflow(
    workflow_data: Dict[str, Any],
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None
) -> Dict[str, Any]:
    """Saves or updates a workflow in the user custom workflows directory."""
    if not isinstance(workflow_data, dict):
        raise WorkflowFormatError("Workflow data must be a valid JSON dictionary.")

    # Validate parseability
    WorkflowParser.parse(workflow_data)

    if name:
        workflow_data["name"] = name
    if description:
        workflow_data["description"] = description
    if author:
        workflow_data["author"] = author

    if not workflow_id:
        base_name = workflow_data.get("name", "custom_workflow")
        import re
        workflow_id = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name).strip("_").lower()
        if not workflow_id:
            workflow_id = f"workflow_{int(time.time())}"

    user_dir = get_user_workflows_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    target_path = user_dir / f"{workflow_id}.json"

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(workflow_data, f, indent=2, ensure_ascii=False)

    return {
        "id": workflow_id,
        "name": workflow_data.get("name", workflow_id),
        "path": str(target_path)
    }


def update_workflow_node_parameter(
    workflow_data: Dict[str, Any],
    node_id: Union[str, int],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Updates parameters for a specific node in either UI format or API format."""
    node_id_str = str(node_id)
    fmt = WorkflowParser.detect_format(workflow_data)

    if fmt == "UI_GRAPH":
        nodes = workflow_data.get("nodes", [])
        target_node = None
        for n in nodes:
            if str(n.get("id")) == node_id_str:
                target_node = n
                break
        if not target_node:
            for sg in workflow_data.get("definitions", {}).get("subgraphs", []):
                for sn in sg.get("nodes", []):
                    if str(sn.get("id")) == node_id_str:
                        target_node = sn
                        break
                if target_node:
                    break

        if not target_node:
            raise WorkflowError(f"Node #{node_id_str} not found in UI workflow.")

        ctype_lower = str(target_node.get("type", "")).lower()
        wvals = list(target_node.get("widgets_values", []))

        # Update specific parameters based on node class
        if "cliptextencode" in ctype_lower:
            if "text" in params:
                if len(wvals) >= 1:
                    wvals[0] = str(params["text"])
                else:
                    wvals.append(str(params["text"]))
        elif "ksampler" in ctype_lower:
            while len(wvals) < 7:
                wvals.append(None)
            if "seed" in params:
                wvals[0] = int(params["seed"])
            if "steps" in params:
                wvals[2] = int(params["steps"])
            if "cfg" in params:
                wvals[3] = float(params["cfg"])
            if "sampler_name" in params:
                wvals[4] = str(params["sampler_name"])
            if "scheduler" in params:
                wvals[5] = str(params["scheduler"])
            if "denoise" in params:
                wvals[6] = float(params["denoise"])
        elif "emptylatentimage" in ctype_lower or "emptysd3latentimage" in ctype_lower:
            while len(wvals) < 3:
                wvals.append(None)
            if "width" in params:
                wvals[0] = int(params["width"])
            if "height" in params:
                wvals[1] = int(params["height"])
            if "batch_size" in params:
                wvals[2] = int(params["batch_size"])
        elif "facedetailer" in ctype_lower:
            while len(wvals) < 8:
                wvals.append(None)
            if "guide_size" in params:
                wvals[0] = int(params["guide_size"])
            if "bbox_threshold" in params:
                wvals[1] = float(params["bbox_threshold"])
            if "steps" in params:
                wvals[3] = int(params["steps"])
            if "cfg" in params:
                wvals[4] = float(params["cfg"])
            if "denoise" in params:
                wvals[5] = float(params["denoise"])
        elif "checkpointloadersimple" in ctype_lower or "unetloader" in ctype_lower or "cliploader" in ctype_lower or "vaeloader" in ctype_lower:
            for k in ("ckpt_name", "unet_name", "clip_name", "vae_name"):
                if k in params:
                    if len(wvals) >= 1:
                        wvals[0] = str(params[k])
                    else:
                        wvals.append(str(params[k]))
                    break
        elif "resolutionselector" in ctype_lower:
            if "aspect_ratio" in params:
                if len(wvals) >= 1:
                    wvals[0] = str(params["aspect_ratio"])
                else:
                    wvals.append(str(params["aspect_ratio"]))
        else:
            if "widgets_values" in params and isinstance(params["widgets_values"], list):
                wvals = params["widgets_values"]

        target_node["widgets_values"] = wvals
        if "widgets_values_named" in target_node and isinstance(target_node["widgets_values_named"], dict):
            for k, v in params.items():
                if k != "widgets_values":
                    target_node["widgets_values_named"][k] = v

    elif fmt == "API_PROMPT":
        if node_id_str not in workflow_data:
            raise WorkflowError(f"Node #{node_id_str} not found in API prompt workflow.")
        node_entry = workflow_data[node_id_str]
        inputs = node_entry.setdefault("inputs", {})
        for k, v in params.items():
            if k == "widgets_values":
                continue
            inputs[k] = v

    return workflow_data


def inspect_workflow_details(workflow_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects a workflow and returns full DAG topology, node parameters, and wiring details."""
    raw = WorkflowParser.load_json(workflow_source) if not isinstance(workflow_source, dict) else workflow_source
    fmt = WorkflowParser.detect_format(raw)
    if fmt == "UI_GRAPH":
        raw = WorkflowParser.inline_subgraphs(raw)
    nodes = WorkflowParser.parse(raw)
    order = DAGEngine.topological_sort(nodes)

    titles = {nid: (n.title or n.class_type) for nid, n in nodes.items()}

    edges: List[Dict[str, Any]] = []
    outgoing_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    incoming_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for dst_id, dst_node in nodes.items():
        for in_name, in_val in dst_node.inputs.items():
            if isinstance(in_val, NodeConnection):
                src_id = in_val.source_node_id
                src_slot = in_val.source_slot_index
                edge = {
                    "from_node": src_id,
                    "from_title": titles.get(src_id, src_id),
                    "from_slot": src_slot,
                    "to_node": dst_id,
                    "to_title": titles.get(dst_id, dst_id),
                    "to_slot": in_name,
                    "type": "LINK"
                }
                edges.append(edge)
                incoming_map[dst_id].append({
                    "slot": in_name,
                    "from_node": src_id,
                    "from_title": titles.get(src_id, src_id),
                    "from_slot": src_slot
                })
                outgoing_map[src_id].append({
                    "to_node": dst_id,
                    "to_title": titles.get(dst_id, dst_id),
                    "slot": in_name
                })

    ROLE_MAP = {
        "checkpointloadersimple": "Loads neural network weights (UNet/DiT, CLIP, VAE) into device memory.",
        "unetloader": "Loads standalone UNet / DiT diffusion backbone into compute graph.",
        "cliploader": "Loads standalone CLIP / Multimodal text encoder weights into memory.",
        "dualcliploader": "Loads dual text encoders for SDXL / DiT models.",
        "vaeloader": "Loads standalone Variational Autoencoder (VAE) weights.",
        "loadcheckpoint": "Loads bundled checkpoint weights.",
        "diffusionmodel": "Loads diffusion backbone transformer weights.",
        "emptylatentimage": "Allocates empty latent noise tensor for diffusion sampling.",
        "emptysdxllatentimage": "Allocates SDXL-resolution empty latent noise tensor.",
        "emptysd3latentimage": "Allocates SD3 / DiT latent noise tensor for diffusion sampling.",
        "resolutionselector": "Calculates optimal canvas dimensions from aspect ratio and megapixels.",
        "modelsamplingauraflow": "Configures AuraFlow / Flux DiT sampling trajectory shift.",
        "modelsamplingflux": "Configures Flux DiT sampling trajectory shift.",
        "cliptextencode": "Encodes natural language prompt into conditioning vector embeddings.",
        "cliptextencodesdxl": "Encodes SDXL dual-text prompt into conditioning vector embeddings.",
        "conditioningzeroout": "Zeroes out text conditioning vector for CFG=1 distillation and speed optimization.",
        "ksampler": "Iteratively denoises latent canvas using numerical solver.",
        "ksampleradvanced": "Advanced multi-stage latent denoising with custom start/end step ranges.",
        "samplercustomadvanced": "Custom sampler pipeline with decoupled scheduler and step controllers.",
        "vaedecode": "Decodes compressed latent space into full-resolution RGB pixel image.",
        "vaedecodetiled": "Tiled VAE decoding with reduced peak VRAM consumption.",
        "facedetailer": "Detects facial bounding boxes, crops, and refines high-frequency features.",
        "facedetailerpipe": "Pipelined face detection and localized 2-step Img2Img refinement.",
        "detailerforeach": "Iterative multi-face localized crop and refinement loop.",
        "saveimage": "Encodes final RGB tensor and exports PNG artifact to gallery.",
        "saveimageadvanced": "Exports output images with advanced format and color space metadata.",
        "previewimage": "Previews in-flight image tensor in web viewport.",
        "markdownnote": "Displays workflow documentation, architectural instructions, and model links.",
        "note": "Informational workflow annotation note."
    }

    steps_info: List[Dict[str, Any]] = []
    nodes_detail: Dict[str, Dict[str, Any]] = {}

    for idx, nid in enumerate(order, 1):
        n = nodes[nid]
        ctype_lower = n.class_type.lower()
        role = ROLE_MAP.get(ctype_lower, f"Executes {n.class_type} pipeline stage.")

        # Extract resolved parameters
        params: Dict[str, Any] = {}
        for k, v in n.inputs.items():
            if not isinstance(v, NodeConnection):
                params[k] = v

        if n.widgets_values:
            if ctype_lower in ("checkpointloadersimple", "unetloader", "cliploader", "vaeloader", "dualcliploader") and len(n.widgets_values) >= 1:
                params.setdefault("ckpt_name", n.widgets_values[0])
            elif ctype_lower in ("resolutionselector",) and len(n.widgets_values) >= 1:
                params.setdefault("aspect_ratio", n.widgets_values[0])
            elif ctype_lower in ("modelsamplingauraflow", "modelsamplingflux") and len(n.widgets_values) >= 1:
                params.setdefault("shift", n.widgets_values[0])
            elif ctype_lower in ("markdownnote", "note") and len(n.widgets_values) >= 1:
                params.setdefault("text", n.widgets_values[0])
            elif ctype_lower in ("emptylatentimage", "emptysdxllatentimage", "emptysd3latentimage") and len(n.widgets_values) >= 2:
                params.setdefault("width", n.widgets_values[0])
                params.setdefault("height", n.widgets_values[1])
                if len(n.widgets_values) >= 3:
                    params.setdefault("batch_size", n.widgets_values[2])
            elif ctype_lower in ("cliptextencode", "cliptextencodesdxl") and len(n.widgets_values) >= 1:
                params.setdefault("text", n.widgets_values[0])
            elif "ksampler" in ctype_lower and len(n.widgets_values) >= 6:
                params.setdefault("seed", n.widgets_values[0])
                params.setdefault("steps", n.widgets_values[2])
                params.setdefault("cfg", n.widgets_values[3])
                params.setdefault("sampler_name", n.widgets_values[4])
                params.setdefault("scheduler", n.widgets_values[5])
                if len(n.widgets_values) >= 7:
                    params.setdefault("denoise", n.widgets_values[6])
            elif "detailer" in ctype_lower and len(n.widgets_values) >= 6:
                params.setdefault("guide_size", n.widgets_values[0])
                params.setdefault("bbox_threshold", n.widgets_values[1])
                params.setdefault("steps", n.widgets_values[3])
                params.setdefault("denoise", n.widgets_values[5])
            elif ctype_lower in ("saveimage", "previewimage", "saveimageadvanced") and len(n.widgets_values) >= 1:
                params.setdefault("filename_prefix", n.widgets_values[0])

        node_info = {
            "step": idx,
            "node_id": nid,
            "class_type": n.class_type,
            "title": n.title or n.class_type,
            "role": role,
            "parameters": params,
            "widgets_values": n.widgets_values,
            "connections_in": incoming_map.get(nid, []),
            "connections_out": outgoing_map.get(nid, [])
        }
        steps_info.append(node_info)
        nodes_detail[nid] = node_info

    return {
        "format": fmt,
        "total_nodes": len(nodes),
        "execution_order": steps_info,
        "nodes_detail": nodes_detail,
        "edges": edges
    }


