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
            return cls._parse_ui_graph(data)
        elif fmt == "API_PROMPT":
            prompt_data = data.get("prompt", data)
            return cls._parse_api_prompt(prompt_data)
        raise WorkflowFormatError(f"Unsupported format: {fmt}")

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
            # Wire up connected input slots
            for inp_slot in n.get("inputs", []):
                inp_name = inp_slot.get("name")
                link_id = inp_slot.get("link")
                if link_id is not None and link_id in link_map:
                    from_node, from_slot, _, _, _ = link_map[link_id]
                    inputs[inp_name] = NodeConnection(source_node_id=from_node, source_slot_index=from_slot)

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
    "unetloader",
    "vaeloader",
    "loadcheckpoint"
)
def handle_checkpoint_loader(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Resolves and loads model weights (GGUF or Safetensors)."""
    ckpt_name = None
    if "ckpt_name" in inputs:
        ckpt_name = str(inputs["ckpt_name"])
    elif node.widgets_values:
        ckpt_name = str(node.widgets_values[0])

    logger.info("[Workflow] [Node %s: %s] Checkpoint resolved: %s", node.id, node.class_type, ckpt_name)
    return {
        "model": ckpt_name or "realistic",
        "clip": ckpt_name or "realistic",
        "vae": ckpt_name or "realistic"
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


@NODE_REGISTRY.register("emptylatentimage", "emptysd3latentimage")
def handle_empty_latent(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Sets initial canvas dimensions."""
    width = 512
    height = 512
    batch_size = 1

    if "width" in inputs:
        width = int(inputs["width"])
    elif len(node.widgets_values) >= 1:
        width = int(node.widgets_values[0])

    if "height" in inputs:
        height = int(inputs["height"])
    elif len(node.widgets_values) >= 2:
        height = int(node.widgets_values[1])

    if "batch_size" in inputs:
        batch_size = int(inputs["batch_size"])
    elif len(node.widgets_values) >= 3:
        batch_size = int(node.widgets_values[2])

    logger.info("[Workflow] [Node %s: %s] Canvas: %dx%d (Batch: %d)", node.id, node.class_type, width, height, batch_size)
    return {"latent": {"width": width, "height": height, "batch_size": batch_size}}


@NODE_REGISTRY.register("ksampler", "ksampleradvanced", "samplercustomadvanced")
def handle_ksampler(node: WorkflowNode, inputs: Dict[str, Any], ctx: ExecutionContext) -> Dict[str, Any]:
    """Coordinates and executes the primary diffusion inference step."""
    from .core import generate

    # Extract model
    model_obj = inputs.get("model", {})
    model_name = model_obj.get("model", "realistic") if isinstance(model_obj, dict) else str(model_obj)

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
    scheduler = inputs.get("scheduler", "flux" if "z_image" in str(model_name).lower() else "karras")
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
        output=str(out_file),
        progress=ctx.progress,
        json_progress=ctx.json_progress
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


@NODE_REGISTRY.register("saveimage", "previewimage")
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

        # 4. Sequential execution loop
        for node_id in exec_order:
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
                    # If dict output, check slot or match by type
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

            # Execute node
            logger.debug("[WorkflowExecutor] Running node #%s: %s", node.id, node.class_type)
            result = handler(node, resolved_inputs, ctx)
            ctx.node_outputs[node.id] = result
            executed_node_ids.append(node.id)

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
        extra_params=kwargs
    )
