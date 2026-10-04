"""AmfyUI Mobile Studio & Local HTTP Daemon for termux-diffusion.

Tailored specifically for smartphone viewports (e.g. Samsung Galaxy S25 / S21 / S20, 360px-420px width).
Zero third-party dependencies: Built strictly with Python standard library (http.server, threading).

Features:
1. Quick Mode: Single-touch generation with model presets, prompt chips, and precision parameter controls.
2. AmfyUI Workflow Engine: ComfyUI UI Graph and API Prompt JSON DAG execution and topological inspection.
3. Real-Time Telemetry: Live progress bar, step counter, speed, ETA, GPU load, and memory gauge.
4. Direct Android Gallery synchronization and high-resolution image preview.
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .core import generate, get_galaxy_gallery_dir, ProgressInfo
from .hub import list_presets, list_cached_models, get_device_model_catalog
from .hardware import (
    detect_hardware_profile,
    get_kernel_physical_memory_gb,
    get_system_resource_metrics,
    TermuxExecutionGuard,
)
from .platform import is_android_termux
from .workflow import (
    WorkflowExecutor,
    WorkflowParser,
    DAGEngine,
    WorkflowError,
    list_builtin_workflows,
    get_builtin_workflow,
    inspect_workflow_details,
    save_user_workflow,
    update_workflow_node_parameter
)

logger = logging.getLogger("termux_diffusion.web")
if not logger.handlers:
    _sh = logging.StreamHandler(sys.stderr)
    _sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(_sh)
    logger.setLevel(logging.INFO)

# Global generation and queue state manager
_STATE_LOCK = threading.Lock()
_QUEUE_LOCK = threading.Lock()
_JOB_QUEUE: List[Dict[str, Any]] = []

_CURRENT_JOB: Dict[str, Any] = {
    "status": "idle",       # "idle", "running", "complete", "error"
    "job_id": None,
    "job_type": None,       # "quick", "workflow"
    "job_name": None,
    "progress": {
        "step": 0,
        "total_steps": 0,
        "percent": 0.0,
        "phase": "idle",
        "speed_s_per_it": None,
        "eta_seconds": None,
        "gpu_busy": None,
        "rss_mb": 0.0,
        "message": "Ready",
        "current_node": None,
        "executed_nodes": []
    },
    "result": None,
    "error": None,
    "start_time": 0.0,
    "elapsed": 0.0
}


def _get_latest_existing_result() -> Optional[Dict[str, Any]]:
    output_dirs = [
        get_galaxy_gallery_dir(),
        Path("/sdcard/DCIM/TermuxDiffusion"),
        Path.home() / "storage" / "dcim" / "TermuxDiffusion",
        Path("/sdcard/Pictures/TermuxDiffusion"),
        Path.home() / "storage" / "pictures" / "TermuxDiffusion",
        Path.home() / ".cache" / "termux-diffusion" / "outputs",
    ]
    latest_file: Optional[Path] = None
    latest_mtime: float = 0.0
    for d in output_dirs:
        try:
            if d.is_dir():
                for p in d.glob("*.png"):
                    try:
                        mtime = p.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_file = p
                    except OSError:
                        pass
        except Exception:
            pass
    if latest_file:
        return {
            "filename": latest_file.name,
            "url": f"/outputs/{latest_file.name}",
            "elapsed_sec": 0.0,
            "device": "vulkan",
            "steps": 0,
            "seed": -1,
            "width": 512,
            "height": 512
        }
    return None


def _update_progress(info: ProgressInfo) -> None:
    with _STATE_LOCK:
        _CURRENT_JOB["progress"]["step"] = info.step
        _CURRENT_JOB["progress"]["total_steps"] = info.total_steps
        _CURRENT_JOB["progress"]["percent"] = info.percent
        _CURRENT_JOB["progress"]["phase"] = info.phase
        _CURRENT_JOB["progress"]["speed_s_per_it"] = info.speed_s_per_it
        _CURRENT_JOB["progress"]["eta_seconds"] = info.eta_seconds
        _CURRENT_JOB["progress"]["gpu_busy"] = info.gpu_busy
        _CURRENT_JOB["progress"]["rss_mb"] = info.rss_mb
        _CURRENT_JOB["progress"]["message"] = info.raw_message or info.phase


def _get_queue_snapshot() -> Dict[str, Any]:
    with _QUEUE_LOCK:
        items = [
            {"id": j["id"], "name": j.get("name", "Job"), "type": j.get("type", "quick"), "enqueued_at": j.get("enqueued_at", 0)}
            for j in _JOB_QUEUE
        ]
        return {"total": len(items), "items": items}


def _enqueue_job(job_type: str, payload: Dict[str, Any], name: str = "") -> Dict[str, Any]:
    job_id = f"job_{int(time.time() * 1000)}_{len(_JOB_QUEUE) + 1}"
    item = {
        "id": job_id,
        "type": job_type,
        "name": name or f"{job_type.capitalize()} Task",
        "payload": payload,
        "enqueued_at": time.time()
    }
    with _QUEUE_LOCK:
        _JOB_QUEUE.append(item)
        pos = len(_JOB_QUEUE)
    logger.info("[JobQueue] Enqueued %s (ID: %s, position: %d)", item["name"], job_id, pos)
    return {"job_id": job_id, "queue_position": pos, "queue_total": pos}


def _remove_job_from_queue(job_id: str) -> bool:
    with _QUEUE_LOCK:
        for idx, j in enumerate(_JOB_QUEUE):
            if j["id"] == job_id:
                _JOB_QUEUE.pop(idx)
                logger.info("[JobQueue] Cancelled job ID %s", job_id)
                return True
    return False


def _clear_job_queue() -> None:
    with _QUEUE_LOCK:
        count = len(_JOB_QUEUE)
        _JOB_QUEUE.clear()
        logger.info("[JobQueue] Cleared %d jobs from queue", count)


def _process_next_job_in_pipeline() -> None:
    with _STATE_LOCK:
        if _CURRENT_JOB["status"] == "running":
            return

    next_job = None
    with _QUEUE_LOCK:
        if _JOB_QUEUE:
            next_job = _JOB_QUEUE.pop(0)

    if not next_job:
        return

    logger.info("[JobQueue] Dequeuing next pipeline task: %s (ID: %s)", next_job["name"], next_job["id"])
    if next_job["type"] == "quick":
        _start_quick_generation(next_job["payload"], job_id=next_job["id"], job_name=next_job["name"])
    else:
        _start_workflow_execution(next_job["payload"], job_id=next_job["id"], job_name=next_job["name"])


def _start_quick_generation(body: Dict[str, Any], job_id: Optional[str] = None, job_name: Optional[str] = None) -> None:
    with _STATE_LOCK:
        _CURRENT_JOB["status"] = "running"
        _CURRENT_JOB["job_id"] = job_id or f"job_{int(time.time() * 1000)}"
        _CURRENT_JOB["job_type"] = "quick"
        _CURRENT_JOB["job_name"] = job_name or f"Quick: {body.get('model', 'realistic')}"
        _CURRENT_JOB["error"] = None
        _CURRENT_JOB["result"] = None
        _CURRENT_JOB["start_time"] = time.time()
        _CURRENT_JOB["elapsed"] = 0.0
        _CURRENT_JOB["progress"] = {
            "step": 0, "total_steps": 1, "percent": 0.0, "phase": "init",
            "speed_s_per_it": None, "eta_seconds": None, "gpu_busy": None, "rss_mb": 0.0,
            "message": "Initializing pipeline...",
            "current_node": {"id": "1", "title": "Direct Model Inference", "class_type": "Txt2Img", "index": 1, "total": 1},
            "executed_nodes": []
        }

    def worker():
        try:
            # Web UI default: Auto-protect with Safe Mode (GPU 90% cap, VAE tiling, Zero-LMK) unless explicitly disabled
            raw_safe = body.get("safe_mode", True)
            safe_mode = raw_safe.lower() not in ("false", "0", "off", "no") if isinstance(raw_safe, str) else bool(raw_safe)
            raw_duty = body.get("gpu_duty_cycle", 90 if safe_mode else None)
            gpu_duty_cycle = int(raw_duty) if raw_duty is not None else (90 if safe_mode else None)

            with TermuxExecutionGuard(enable_wake_lock=True, boost_priority=not safe_mode, safe_mode=safe_mode):
                prompt = body.get("prompt", "")
                model = body.get("model", "realistic")
                device = body.get("device", "auto")
                steps = body.get("steps")
                cfg = body.get("cfg")
                width = int(body.get("width", 512))
                height = int(body.get("height", 512))
                seed = int(body.get("seed", -1))
                detail_faces = bool(body.get("detail_faces", False))
                negative = body.get("negative_prompt")

                res = generate(
                    prompt=prompt,
                    negative_prompt=negative,
                    model=model,
                    device=device,
                    steps=steps,
                    cfg_scale=cfg,
                    width=width,
                    height=height,
                    seed=seed,
                    detail_faces=detail_faces,
                    safe_mode=safe_mode,
                    gpu_duty_cycle=gpu_duty_cycle,
                    progress_callback=_update_progress,
                    progress=True
                )

                with _STATE_LOCK:
                    _CURRENT_JOB["status"] = "complete"
                    _CURRENT_JOB["elapsed"] = res.elapsed_sec
                    _CURRENT_JOB["progress"]["executed_nodes"] = ["1"]
                    _CURRENT_JOB["result"] = {
                        "filename": res.path.name,
                        "url": f"/outputs/{res.path.name}",
                        "elapsed_sec": res.elapsed_sec,
                        "device": res.device,
                        "steps": res.steps,
                        "seed": res.seed,
                        "width": res.width,
                        "height": res.height
                    }
        except Exception as e:
            logger.exception("[Worker] Generation failed: %s", e)
            with _STATE_LOCK:
                _CURRENT_JOB["status"] = "error"
                _CURRENT_JOB["error"] = str(e)
        finally:
            _process_next_job_in_pipeline()

    t = threading.Thread(target=worker, daemon=True, name="DiffusionWorker")
    t.start()


def _start_workflow_execution(body: Dict[str, Any], job_id: Optional[str] = None, job_name: Optional[str] = None) -> None:
    with _STATE_LOCK:
        _CURRENT_JOB["status"] = "running"
        _CURRENT_JOB["job_id"] = job_id or f"job_{int(time.time() * 1000)}"
        _CURRENT_JOB["job_type"] = "workflow"
        wf_name = (body.get("workflow", {}) if isinstance(body.get("workflow"), dict) else {}).get("name", "Custom Workflow")
        _CURRENT_JOB["job_name"] = job_name or f"AmfyUI: {wf_name}"
        _CURRENT_JOB["error"] = None
        _CURRENT_JOB["result"] = None
        _CURRENT_JOB["start_time"] = time.time()
        _CURRENT_JOB["elapsed"] = 0.0
        _CURRENT_JOB["progress"] = {
            "step": 0, "total_steps": 1, "percent": 0.0, "phase": "init",
            "speed_s_per_it": None, "eta_seconds": None, "gpu_busy": None, "rss_mb": 0.0,
            "message": "Initializing DAG graph...",
            "current_node": None,
            "executed_nodes": []
        }

    def on_node_stage(stage_info: Dict[str, Any]):
        with _STATE_LOCK:
            _CURRENT_JOB["progress"]["current_node"] = {
                "id": stage_info["node_id"],
                "title": stage_info["node_title"],
                "class_type": stage_info["class_type"],
                "index": stage_info["current_node_index"],
                "total": stage_info["total_nodes"],
                "status": stage_info["status"]
            }
            _CURRENT_JOB["progress"]["phase"] = stage_info["phase"]
            _CURRENT_JOB["progress"]["message"] = stage_info["phase"]
            _CURRENT_JOB["progress"]["percent"] = stage_info["percent"]
            _CURRENT_JOB["progress"]["executed_nodes"] = stage_info["executed_nodes"]

    def wf_worker():
        try:
            raw_safe = body.get("safe_mode", True)
            safe_mode = raw_safe.lower() not in ("false", "0", "off", "no") if isinstance(raw_safe, str) else bool(raw_safe)
            with TermuxExecutionGuard(enable_wake_lock=True, boost_priority=not safe_mode, safe_mode=safe_mode):
                if "workflow_id" in body and "workflow" not in body and "nodes" not in body:
                    wf_id = str(body["workflow_id"])
                    wf_data = get_builtin_workflow(wf_id)
                    if not wf_data:
                        raise WorkflowError(f"Workflow preset '{wf_id}' not found")
                else:
                    wf_data = body.get("workflow", body)
                device = body.get("device", "auto")
                force_run = bool(body.get("force_run", True))
                out_dir = Path.home() / ".cache" / "termux-diffusion" / "outputs"
                out_dir.mkdir(parents=True, exist_ok=True)

                executor = WorkflowExecutor()
                res = executor.run(
                    workflow=wf_data,
                    output_dir=out_dir,
                    device=device,
                    force_run=force_run,
                    progress=True,
                    node_progress_callback=on_node_stage
                )

                with _STATE_LOCK:
                    _CURRENT_JOB["status"] = "complete"
                    _CURRENT_JOB["elapsed"] = res.elapsed_seconds
                    out_urls = [f"/outputs/{p.name}" for p in res.output_images]
                    _CURRENT_JOB["result"] = {
                        "images": out_urls,
                        "url": out_urls[0] if out_urls else None,
                        "filename": res.output_images[0].name if res.output_images else None,
                        "elapsed_sec": res.elapsed_seconds,
                        "executed_nodes": res.executed_nodes
                    }
        except Exception as e:
            logger.exception("[WorkflowWorker] Run failed: %s", e)
            with _STATE_LOCK:
                _CURRENT_JOB["status"] = "error"
                _CURRENT_JOB["error"] = str(e)
        finally:
            _process_next_job_in_pipeline()

    t = threading.Thread(target=wf_worker, daemon=True, name="WorkflowWorker")
    t.start()


class DiffusionWebHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for mobile client and REST APIs."""

    server_version = "TermuxDiffusionWeb/1.4.0"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("[HTTP] " + format % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_mobile_ui()
        elif path == "/api/models":
            self._serve_models_api()
        elif path == "/api/status":
            self._serve_status_api()
        elif path == "/api/workflows":
            self._serve_workflows_api()
        elif path.startswith("/api/workflows/") and path.endswith("/export"):
            parts = path.split("/")
            wf_id = parts[3]
            self._serve_workflow_export_api(wf_id)
        elif path.startswith("/api/workflows/"):
            wf_id = path[len("/api/workflows/"):]
            self._serve_workflow_detail_api(wf_id)
        elif path == "/api/queue":
            with _STATE_LOCK:
                curr = dict(_CURRENT_JOB)
            self._send_json({
                "current_job": curr if curr.get("status") == "running" else None,
                "queue": _get_queue_snapshot()
            })
        elif path == "/api/progress":
            with _STATE_LOCK:
                job_data = dict(_CURRENT_JOB)
                job_data["progress"] = dict(_CURRENT_JOB["progress"])
                if job_data["status"] == "running" and job_data["start_time"] > 0:
                    job_data["elapsed"] = round(time.time() - job_data["start_time"], 2)
                if not job_data.get("result"):
                    job_data["result"] = _get_latest_existing_result()
            try:
                metrics = get_system_resource_metrics()
            except Exception:
                metrics = {
                    "ram_total_gb": 0.0, "ram_avail_gb": 0.0, "ram_used_gb": 0.0,
                    "ram_percent": 0.0, "cpu_percent": 0.0, "gpu_percent": 0.0,
                    "process_rss_mb": 0.0
                }
            job_data["metrics"] = metrics
            job_data["queue"] = _get_queue_snapshot()
            self._send_json(job_data)
        elif path.startswith("/outputs/"):
            self._serve_output_file(path[len("/outputs/"):])
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Malformed JSON payload"}, 400)
            return

        if path == "/api/generate":
            self._handle_api_generate(body)
        elif path == "/api/workflow/inspect":
            self._handle_api_workflow_inspect(body)
        elif path == "/api/workflow/save":
            self._handle_api_workflow_save(body)
        elif path == "/api/workflow/update_node":
            self._handle_api_workflow_update_node(body)
        elif path == "/api/workflow/run":
            self._handle_api_workflow_run(body)
        elif path == "/api/queue/cancel":
            job_id = body.get("job_id")
            if not job_id:
                self._send_json({"error": "Missing job_id"}, 400)
                return
            removed = _remove_job_from_queue(str(job_id))
            self._send_json({"success": removed, "job_id": job_id})
        elif path == "/api/queue/clear":
            _clear_job_queue()
            self._send_json({"success": True, "message": "Queue cleared"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")

    def _serve_models_api(self) -> None:
        presets = list_presets()
        cached = list_cached_models()
        catalog = get_device_model_catalog()
        hw = detect_hardware_profile()
        ram_info = get_kernel_physical_memory_gb()
        ram_val = ram_info[0] if isinstance(ram_info, (tuple, list)) else ram_info

        self._send_json({
            "presets": presets,
            "cached_models": cached,
            "model_catalog": catalog,
            "hardware": {
                "soc": hw.soc_name,
                "gpu": hw.gpu_name,
                "ram_gb": round(float(ram_val), 1),
                "is_termux": is_android_termux()
            }
        })

    def _serve_status_api(self) -> None:
        with _STATE_LOCK:
            job_copy = dict(_CURRENT_JOB)
        ram_info = get_kernel_physical_memory_gb()
        ram_val = ram_info[0] if isinstance(ram_info, (tuple, list)) else ram_info
        self._send_json({
            "job": job_copy,
            "system": {
                "ram_gb": round(float(ram_val), 1),
                "time": time.time()
            }
        })

    def _serve_workflows_api(self) -> None:
        workflows = list_builtin_workflows()
        self._send_json({"workflows": workflows, "total": len(workflows)})

    def _serve_workflow_detail_api(self, wf_id: str) -> None:
        data = get_builtin_workflow(wf_id)
        if not data:
            self.send_error(HTTPStatus.NOT_FOUND, f"Workflow not found: {wf_id}")
            return
        self._send_json(data)

    def _serve_workflow_export_api(self, wf_id: str) -> None:
        data = get_builtin_workflow(wf_id)
        if not data:
            self.send_error(HTTPStatus.NOT_FOUND, f"Workflow not found: {wf_id}")
            return
        content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{wf_id}.json"')
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_output_file(self, filename: str) -> None:
        safe_name = os.path.basename(filename)
        output_dirs = [
            get_galaxy_gallery_dir(),
            Path("/sdcard/DCIM/TermuxDiffusion"),
            Path.home() / "storage" / "dcim" / "TermuxDiffusion",
            Path.home() / ".cache" / "termux-diffusion" / "outputs",
            Path("/sdcard/Pictures/TermuxDiffusion"),
            Path.home() / "storage" / "pictures" / "TermuxDiffusion",
            Path.home(),
            Path.cwd()
        ]
        target_file = None
        for d in output_dirs:
            p = d / safe_name
            if p.is_file():
                target_file = p
                break

        if not target_file:
            self.send_error(HTTPStatus.NOT_FOUND, f"Image not found: {safe_name}")
            return

        mime_type, _ = mimetypes.guess_type(str(target_file))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(target_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed reading file: {e}")

    def _handle_api_generate(self, body: Dict[str, Any]) -> None:
        with _STATE_LOCK:
            is_running = (_CURRENT_JOB["status"] == "running")

        model_name = body.get("model", "realistic")
        prompt_snippet = (body.get("prompt") or "")[:24]
        job_name = f"Quick: {model_name} ('{prompt_snippet}...')"

        if is_running:
            enqueue_info = _enqueue_job("quick", body, name=job_name)
            self._send_json({
                "message": "Task added to pipeline queue",
                "status": "queued",
                "job_id": enqueue_info["job_id"],
                "queue_position": enqueue_info["queue_position"],
                "queue_total": enqueue_info["queue_total"]
            }, 202)
        else:
            _start_quick_generation(body, job_name=job_name)
            self._send_json({"message": "Generation dispatched", "status": "running"})

    def _handle_api_workflow_inspect(self, body: Dict[str, Any]) -> None:
        try:
            if "workflow_id" in body and "workflow" not in body and "nodes" not in body:
                wf_id = str(body["workflow_id"])
                wf_data = get_builtin_workflow(wf_id)
                if not wf_data:
                    self._send_json({"error": f"Workflow preset '{wf_id}' not found"}, 404)
                    return
            else:
                wf_data = body.get("workflow", body)
            inspection_result = inspect_workflow_details(wf_data)
            self._send_json(inspection_result)
        except WorkflowError as wfe:
            self._send_json({"error": str(wfe)}, 400)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_api_workflow_save(self, body: Dict[str, Any]) -> None:
        try:
            wf_data = body.get("workflow")
            if not wf_data:
                self._send_json({"error": "Missing 'workflow' object in payload"}, 400)
                return
            wf_id = body.get("id")
            name = body.get("name")
            desc = body.get("description")
            res = save_user_workflow(wf_data, workflow_id=wf_id, name=name, description=desc)
            self._send_json({"success": True, "saved": res})
        except Exception as e:
            logger.exception("[Web] Failed saving user workflow: %s", e)
            self._send_json({"error": str(e)}, 400)

    def _handle_api_workflow_update_node(self, body: Dict[str, Any]) -> None:
        try:
            wf_data = body.get("workflow")
            node_id = body.get("node_id")
            params = body.get("parameters", {})
            if not wf_data or node_id is None:
                self._send_json({"error": "Missing 'workflow' or 'node_id' in payload"}, 400)
                return
            updated_wf = update_workflow_node_parameter(wf_data, node_id, params)
            inspection = inspect_workflow_details(updated_wf)
            self._send_json({"success": True, "workflow": updated_wf, "inspection": inspection})
        except Exception as e:
            logger.exception("[Web] Failed updating node parameter: %s", e)
            self._send_json({"error": str(e)}, 400)

    def _handle_api_workflow_run(self, body: Dict[str, Any]) -> None:
        with _STATE_LOCK:
            is_running = (_CURRENT_JOB["status"] == "running")

        wf_name = "Workflow"
        if "workflow_id" in body and "workflow" not in body:
            wf_name = f"Preset: {body['workflow_id']}"
        elif isinstance(body.get("workflow"), dict):
            wf_name = body["workflow"].get("name", "Custom Workflow")
        job_name = f"AmfyUI: {wf_name}"

        if is_running:
            enqueue_info = _enqueue_job("workflow", body, name=job_name)
            self._send_json({
                "message": "Workflow added to pipeline queue",
                "status": "queued",
                "job_id": enqueue_info["job_id"],
                "queue_position": enqueue_info["queue_position"],
                "queue_total": enqueue_info["queue_total"]
            }, 202)
        else:
            _start_workflow_execution(body, job_name=job_name)
            self._send_json({"message": "Workflow dispatched", "status": "running"})

    def _serve_mobile_ui(self) -> None:
        """Renders the high-performance, responsive Galaxy S25 OLED dark mobile interface."""
        html = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>AMEVA AmfyUI Studio</title>
  <style>
    :root {
      --bg: #07090e;
      --card: #111622;
      --card-border: #1e293b;
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --success: #10b981;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --radius: 16px;
      --safe-bottom: env(safe-area-inset-bottom, 16px);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Pretendard", "Segoe UI", Roboto, sans-serif;
      font-size: 15px;
      line-height: 1.4;
      padding-bottom: calc(90px + var(--safe-bottom));
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 16px 20px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--card-border);
      background: rgba(11, 15, 23, 0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 17px; }
    .brand-badge {
      font-size: 10px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: #fff;
      padding: 2px 7px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .status-badge {
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 12px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
    .status-badge.running { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
    .status-badge.running .status-dot { animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { opacity: 0.3; } to { opacity: 1; } }

    /* Resource HUD Bar */
    .resource-hud {
      display: flex;
      gap: 6px;
      padding: 7px 16px;
      background: #090e1a;
      border-bottom: 1px solid var(--card-border);
      overflow-x: auto;
      white-space: nowrap;
      scrollbar-width: none;
      align-items: center;
      position: sticky;
      top: 57px;
      z-index: 45;
      backdrop-filter: blur(8px);
    }
    .resource-hud::-webkit-scrollbar { display: none; }
    .hud-chip {
      background: #111827;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 4px 8px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      font-family: -apple-system, BlinkMacSystemFont, monospace;
      color: var(--text);
    }
    .hud-label {
      font-size: 9px;
      font-weight: 800;
      color: var(--text-muted);
      letter-spacing: 0.5px;
    }
    .hud-val {
      font-weight: 700;
      color: #38bdf8;
    }
    .hud-chip.queue-chip {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .hud-chip.queue-chip:hover {
      background: rgba(99, 102, 241, 0.25);
    }
    .hud-chip.queue-chip .hud-val {
      color: #a5b4fc;
    }

    /* Live Node Execution States */
    .flow-node-card.flow-node-executing {
      border-color: #6366f1 !important;
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.8), inset 0 0 8px rgba(99, 102, 241, 0.3) !important;
      animation: executingPulse 1.2s infinite alternate ease-in-out;
      position: relative;
    }
    @keyframes executingPulse {
      from { border-color: #6366f1; box-shadow: 0 0 10px rgba(99, 102, 241, 0.5); }
      to { border-color: #a855f7; box-shadow: 0 0 22px rgba(168, 85, 247, 0.9); }
    }
    .flow-node-card.flow-node-executing::after {
      content: "▶ RUNNING";
      position: absolute;
      top: -8px; right: 10px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      font-size: 8px;
      font-weight: 900;
      padding: 1px 6px;
      border-radius: 4px;
      letter-spacing: 0.5px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .flow-node-card.flow-node-completed {
      border-color: #10b981 !important;
      position: relative;
    }
    .flow-node-card.flow-node-completed::after {
      content: "✓ DONE";
      position: absolute;
      top: -8px; right: 10px;
      background: #10b981;
      color: white;
      font-size: 8px;
      font-weight: 900;
      padding: 1px 6px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }
    .generate-btn.enqueue-mode {
      background: linear-gradient(135deg, #4f46e5, #0891b2) !important;
      box-shadow: 0 4px 18px rgba(8, 145, 178, 0.45) !important;
    }
    .queue-item {
      background: #141c2e;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }

    main { padding: 16px; flex: 1; max-width: 480px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; gap: 16px; }

    /* Segmented Tab Switcher */
    .tabs {
      display: flex;
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 4px;
      gap: 4px;
    }
    .tab-btn {
      flex: 1;
      padding: 10px 0;
      font-size: 14px;
      font-weight: 600;
      border: none;
      background: transparent;
      color: var(--text-muted);
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .tab-btn.active {
      background: #1e293b;
      color: var(--text);
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Cards */
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
    }

    /* Model Chips */
    .chip-scroll {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
      scrollbar-width: none;
    }
    .chip-scroll::-webkit-scrollbar { display: none; }
    .chip {
      flex-shrink: 0;
      padding: 7px 14px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 20px;
      background: #182234;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .chip.active {
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      border-color: #6366f1;
      color: #fff;
      box-shadow: 0 2px 10px var(--accent-glow);
    }

    /* Inputs */
    textarea, input[type="text"] {
      width: 100%;
      background: #0b0f19;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      color: var(--text);
      font-size: 15px;
      font-family: inherit;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus, input[type="text"]:focus {
      border-color: var(--accent);
    }
    .quick-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
    .tag {
      font-size: 11px;
      padding: 3px 9px;
      border-radius: 8px;
      background: #182234;
      color: #94a3b8;
      cursor: pointer;
    }

    /* Controls Grid */
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .ctrl-group { display: flex; flex-direction: column; gap: 6px; }
    .ctrl-group label {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    input[type="range"] {
      accent-color: var(--accent);
      cursor: pointer;
    }

    /* Toggle Switch */
    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 4px 0;
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background-color: #1e293b; transition: .2s; border-radius: 24px;
    }
    .slider:before {
      position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px;
      background-color: white; transition: .2s; border-radius: 50%;
    }
    input:checked + .slider { background-color: var(--accent); }
    input:checked + .slider:before { transform: translateX(20px); }

    /* Progress Display */
    .progress-box {
      display: none;
      background: #111726;
      border: 1px solid var(--accent);
      border-radius: var(--radius);
      padding: 16px;
      flex-direction: column;
      gap: 10px;
    }
    .progress-box.active { display: flex; }
    .bar-bg {
      width: 100%;
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #6366f1, #38bdf8);
      transition: width 0.3s ease;
    }
    .metrics-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 600;
    }

    /* Image Preview */
    .preview-box {
      display: none;
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 12px;
      flex-direction: column;
      gap: 12px;
      align-items: center;
    }
    .preview-box.active { display: flex; }
    .preview-img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 12px;
      background: #0b0f19;
    }
    .preview-actions { display: flex; gap: 8px; width: 100%; }
    .action-btn {
      flex: 1;
      padding: 10px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 10px;
      border: 1px solid var(--card-border);
      background: #1e293b;
      color: var(--text);
      cursor: pointer;
      text-align: center;
      text-decoration: none;
    }

    /* Fixed Bottom Action Bar */
    .bottom-bar {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(11, 15, 23, 0.9);
      backdrop-filter: blur(16px);
      padding: 12px 16px calc(12px + var(--safe-bottom));
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: center;
      z-index: 100;
    }
    .generate-btn {
      width: 100%;
      max-width: 480px;
      padding: 16px;
      font-size: 17px;
      font-weight: 700;
      border-radius: 16px;
      border: none;
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: #fff;
      cursor: pointer;
      box-shadow: 0 4px 20px var(--accent-glow);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: transform 0.1s, opacity 0.2s;
    }
    .generate-btn:active { transform: scale(0.98); }
    .generate-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      background: #334155;
      box-shadow: none;
    }

    /* Workflow Tab Drop Zone */
    .drop-zone {
      border: 2px dashed var(--card-border);
      border-radius: 14px;
      padding: 24px 16px;
      text-align: center;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .drop-zone:hover { border-color: var(--accent); }
    .node-sequence { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
    .node-pill {
      background: #182234;
      padding: 8px 12px;
      border-radius: 10px;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .node-num {
      width: 20px;
      height: 20px;
      background: var(--accent);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 700;
    }
    .wf-preset-list { display: flex; flex-direction: column; gap: 8px; }
    .wf-card {
      background: #0b0f19;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .wf-card:hover { border-color: rgba(99, 102, 241, 0.6); }
    .wf-card.active {
      border-color: var(--accent);
      background: #151a2d;
      box-shadow: 0 2px 12px var(--accent-glow);
    }
    .wf-card-header { display: flex; justify-content: space-between; align-items: center; }
    .wf-card-title { font-size: 14px; font-weight: 700; color: var(--text); }
    .wf-card-badge {
      font-size: 10px;
      font-weight: 600;
      background: #1e293b;
      color: #94a3b8;
      padding: 2px 8px;
      border-radius: 10px;
    }
    .wf-card.active .wf-card-badge { background: var(--accent); color: #fff; }
    .wf-card-desc { font-size: 12px; color: var(--text-muted); line-height: 1.3; }

    /* Visual DAG Flow Diagram */
    .dag-view-switcher {
      display: flex;
      gap: 6px;
      margin-top: 6px;
      margin-bottom: 8px;
    }
    .dag-view-btn {
      flex: 1;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      background: #111622;
      color: var(--text-muted);
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }
    .dag-view-btn.active {
      background: #1e293b;
      color: #fff;
      border-color: var(--accent);
    }

    .dag-flow-container {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 6px;
    }
    .flow-node-card {
      background: #0f1626;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .flow-node-card:hover {
      border-color: rgba(99, 102, 241, 0.7);
      background: #141d33;
    }
    .flow-node-card.selected {
      border-color: var(--accent);
      background: #18223c;
      box-shadow: 0 0 16px var(--accent-glow);
    }
    .flow-node-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .flow-node-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .flow-node-badge {
      font-size: 9px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 6px;
      background: #1e293b;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .flow-node-badge.loader { background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .flow-node-badge.conditioning { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .flow-node-badge.sampler { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .flow-node-badge.decode { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .flow-node-badge.detailer { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
    .flow-node-badge.save { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

    .flow-node-params {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }
    .flow-param-chip {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      background: #1e293b;
      color: #cbd5e1;
      font-family: monospace;
    }
    .flow-arrow-down {
      display: flex;
      justify-content: center;
      color: var(--accent);
      font-size: 13px;
      opacity: 0.5;
    }

    /* Node Inspector Panel */
    .inspector-card {
      display: none;
      background: #0a0f1d;
      border: 1px solid var(--accent);
      border-radius: var(--radius);
      padding: 14px;
      flex-direction: column;
      gap: 10px;
      box-shadow: 0 4px 20px var(--accent-glow);
      animation: fadeIn 0.2s ease;
    }
    .inspector-card.active { display: flex; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .inspector-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
    }
    .inspector-role {
      font-size: 12px;
      color: #cbd5e1;
      line-height: 1.4;
      background: #141b2e;
      padding: 8px 10px;
      border-radius: 8px;
      border-left: 3px solid var(--accent);
    }
    .inspector-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .inspector-prop {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 6px 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .inspector-prop-key {
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .inspector-prop-val {
      font-size: 12px;
      font-weight: 600;
      color: #38bdf8;
      font-family: monospace;
      word-break: break-all;
    }
    .inspector-prompt-box {
      background: #070c16;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
      color: #f8fafc;
      line-height: 1.4;
      max-height: 100px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .wiring-box {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .wiring-row {
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 6px;
      background: #141c2e;
      border: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: monospace;
    }
    .wiring-row .slot-name { color: #818cf8; font-weight: 600; }
    .wiring-row .target-node { color: #94a3b8; }

    /* Modal Dialog */
    .modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 16px;
    }
    .modal-card {
      background: #0d1527;
      border: 1px solid var(--accent);
      border-radius: var(--radius);
      padding: 20px;
      width: 100%;
      max-width: 380px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      gap: 12px;
      animation: fadeIn 0.2s ease;
    }
    .modal-title {
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .custom-input, .custom-select, .custom-textarea {
      background: #141d33;
      color: #f8fafc;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 12px;
      width: 100%;
      box-sizing: border-box;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s ease;
    }
    .custom-input:focus, .custom-select:focus, .custom-textarea:focus {
      border-color: var(--accent);
    }
    .inspector-prop.full-span { grid-column: span 2; }
    .param-slider-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 2px;
    }
    .param-slider-wrap input[type="range"] {
      flex: 1;
      height: 6px;
      accent-color: var(--accent);
      background: #1e293b;
      border-radius: 3px;
      outline: none;
    }
    .param-num-compact {
      width: 58px !important;
      text-align: center;
      padding: 4px 6px !important;
      font-weight: 700;
      color: #38bdf8;
      font-family: monospace;
    }
    .custom-path-box {
      display: none;
      margin-top: 6px;
    }
    .model-badge-tag {
      font-size: 9px;
      padding: 1px 6px;
      border-radius: 4px;
      background: rgba(99, 102, 241, 0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.35);
      margin-left: 6px;
      font-weight: 600;
    }

    /* Floating Draggable Bottom Sheet */
    .sheet-overlay {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.45);
      z-index: 800;
      transition: opacity 0.2s ease;
    }
    .sheet-overlay.active { display: block; }

    .bottom-sheet {
      position: fixed;
      left: 0; right: 0; bottom: 0;
      z-index: 850;
      background: #090e1d;
      border-top: 2px solid var(--accent);
      border-radius: 20px 20px 0 0;
      box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.85);
      display: flex;
      flex-direction: column;
      transition: height 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s ease;
      transform: translateY(100%);
      max-width: 600px;
      margin: 0 auto;
    }
    .bottom-sheet.open {
      transform: translateY(0);
    }
    .bottom-sheet.peek {
      height: 68px;
    }
    .bottom-sheet.half {
      height: 48vh;
    }
    .bottom-sheet.full {
      height: 82vh;
    }

    .sheet-handle-area {
      width: 100%;
      padding: 8px 0 4px 0;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: grab;
      touch-action: none;
    }
    .sheet-drag-pill {
      width: 44px;
      height: 5px;
      border-radius: 3px;
      background: #475569;
      transition: background 0.2s;
    }
    .sheet-handle-area:hover .sheet-drag-pill, .sheet-handle-area:active .sheet-drag-pill {
      background: var(--accent);
    }

    .sheet-header-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 16px 8px 16px;
      border-bottom: 1px solid var(--card-border);
    }
    .sheet-header-left {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      overflow: hidden;
    }
    .sheet-header-controls {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .sheet-btn {
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: #cbd5e1;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      cursor: pointer;
      font-weight: 600;
    }
    .sheet-btn:hover { background: #334155; }
    .sheet-btn.close { color: #f87171; }

    .sheet-content {
      flex: 1;
      overflow-y: auto;
      padding: 14px 16px 24px 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    /* Multi-Branch Port Badges */
    .flow-ports-in, .flow-ports-out {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 4px;
      margin-top: 4px;
      font-size: 10px;
    }
    .ports-label {
      font-weight: 700;
      font-size: 9px;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .port-chip {
      padding: 2px 6px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 10px;
      font-weight: 600;
      white-space: nowrap;
    }
    .port-chip.in {
      background: rgba(14, 165, 233, 0.15);
      border: 1px solid rgba(14, 165, 233, 0.35);
      color: #38bdf8;
    }
    .port-chip.out {
      background: rgba(168, 85, 247, 0.15);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: #c084fc;
    }

    /* Multi-Node Branch Lighting & Dimming */
    .flow-node-card.flow-node-source {
      border-color: #38bdf8 !important;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.45) !important;
      position: relative;
    }
    .flow-node-card.flow-node-source::after {
      content: "▲ INFLOW SOURCE";
      position: absolute;
      top: -8px; right: 10px;
      background: #0284c7;
      color: white;
      font-size: 8px;
      font-weight: 800;
      padding: 1px 5px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }
    .flow-node-card.flow-node-target {
      border-color: #f59e0b !important;
      box-shadow: 0 0 14px rgba(245, 158, 11, 0.45) !important;
      position: relative;
    }
    .flow-node-card.flow-node-target::after {
      content: "▼ OUTFLOW TARGET";
      position: absolute;
      top: -8px; right: 10px;
      background: #d97706;
      color: white;
      font-size: 8px;
      font-weight: 800;
      padding: 1px 5px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }
    .flow-node-card.flow-node-dimmed {
      opacity: 0.35;
      filter: grayscale(40%);
    }

    /* Topology Matrix View */
    .dag-matrix-container {
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 10px;
    }
    .matrix-layer {
      background: #0d1526;
      border: 1px dashed var(--card-border);
      border-radius: 10px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .matrix-layer-title {
      font-size: 10px;
      font-weight: 800;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
    }
    .matrix-row-parallel {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px;
    }
    .matrix-convergence-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      font-family: monospace;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(99, 102, 241, 0.2);
      border: 1px solid rgba(99, 102, 241, 0.4);
      color: #a5b4fc;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>AMEVA</span>
      <span class="brand-badge">AmfyUI</span>
    </div>
    <div class="status-badge" id="statusBadge">
      <span class="status-dot"></span>
      <span id="statusText">Ready</span>
    </div>
  </header>
  
  <div class="resource-hud" id="resourceHud">
    <div class="hud-chip" id="hudRamChip">
      <span class="hud-label">RAM</span>
      <span class="hud-val" id="hudRamVal">--/-- GB</span>
    </div>
    <div class="hud-chip" id="hudCpuChip">
      <span class="hud-label">CPU</span>
      <span class="hud-val" id="hudCpuVal">--%</span>
    </div>
    <div class="hud-chip" id="hudGpuChip">
      <span class="hud-label">GPU</span>
      <span class="hud-val" id="hudGpuVal">--%</span>
    </div>
    <div class="hud-chip" id="hudTimeChip">
      <span class="hud-label">ELAPSED</span>
      <span class="hud-val" id="hudTimeVal">0.0s</span>
    </div>
    <div class="hud-chip queue-chip" id="hudQueueChip" onclick="openQueueModal()" title="작업 대기열 보기">
      <span class="hud-label">QUEUE</span>
      <span class="hud-val" id="hudQueueVal">0 pending</span>
    </div>
  </div>

  <main>
    <div class="tabs">
      <button class="tab-btn active" id="tabQuickBtn" onclick="switchTab('quick')">Quick Mode</button>
      <button class="tab-btn" id="tabWfBtn" onclick="switchTab('wf')">AmfyUI</button>
    </div>

    <!-- TAB 1: Quick Generate -->
    <div id="quickTab" style="display: flex; flex-direction: column; gap: 14px;">
      <div class="card">
        <div class="card-title">Model Selection</div>
        <div class="chip-scroll" id="modelChips">
          <div class="chip active" data-model="realistic" onclick="selectModel(this)">Realistic</div>
          <div class="chip" data-model="z-image-turbo" onclick="selectModel(this)">Z-Image Turbo</div>
          <div class="chip" data-model="sdxs" onclick="selectModel(this)">SDXS Fast</div>
          <div class="chip" data-model="anime" onclick="selectModel(this)">Anime</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Prompt</div>
        <textarea id="promptInput" rows="3" placeholder="Input prompt text...">A professional portrait of an East Asian woman in her 20s smiling warmly, radiant eyes, natural skin texture, 8k photography</textarea>
        <div class="quick-tags">
          <span class="tag" onclick="addTag('Korean woman portrait')">+ Korean Portrait</span>
          <span class="tag" onclick="addTag('natural sunlight, soft daylight')">+ Natural Light</span>
          <span class="tag" onclick="addTag('8k masterpiece, raw photo')">+ 8K Masterpiece</span>
          <span class="tag" onclick="addTag('1girl, cute anime masterpiece')">+ Anime Girl</span>
          <span class="tag" onclick="addTag('makoto shinkai style, cloudy sky')">+ Shinkai Sky</span>
          <span class="tag" onclick="addTag('studio ghibli aesthetic, vibrant')">+ Ghibli Style</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Parameters</div>
        <div class="controls-grid">
          <div class="ctrl-group">
            <label>Steps: <span id="stepsVal">15</span></label>
            <input type="range" id="stepsInput" min="1" max="30" value="15" oninput="stepsVal.innerText=this.value">
          </div>
          <div class="ctrl-group">
            <label>CFG: <span id="cfgVal">7.0</span></label>
            <input type="range" id="cfgInput" min="1.0" max="15.0" step="0.5" value="7.0" oninput="cfgVal.innerText=this.value">
          </div>
        </div>

        <div class="switch-row" style="margin-top: 6px;">
          <span style="font-weight: 600; font-size: 14px;">FaceDetailer (안면 복원)</span>
          <label class="switch">
            <input type="checkbox" id="detailerSwitch" checked>
            <span class="slider"></span>
          </label>
        </div>

        <div class="switch-row" style="margin-top: 6px;" title="GPU 90% 상한 제어 및 프로세스 우선순위 조정으로 화면 렌더링(SurfaceFlinger)과 백그라운드 앱 튕김 방지">
          <div>
            <span style="font-weight: 600; font-size: 14px;">Safe Mode (앱/화면 생존 가드)</span>
            <div style="font-size: 11px; color: var(--text-muted);">GPU 90% Cap &amp; Zero-LMK VAE Tiling</div>
          </div>
          <label class="switch">
            <input type="checkbox" id="safeModeSwitch" checked>
            <span class="slider"></span>
          </label>
        </div>
      </div>
    </div>

    <!-- TAB 2: AmfyUI Workflow -->
    <div id="wfTab" style="display: none; flex-direction: column; gap: 14px;">
      <div class="card">
        <div class="card-title">Standard Workflows (인기 표준 프리셋)</div>
        <div class="wf-preset-list" id="wfPresetList">
          <div class="wf-card active" onclick="selectPresetWorkflow('standard_photoreal', this)">
            <div class="wf-card-header">
              <span class="wf-card-title">Standard Photoreal + FaceDetailer</span>
              <span class="wf-card-badge">8 Nodes</span>
            </div>
            <div class="wf-card-desc">전 세계에서 가장 널리 쓰이는 표준 ComfyUI TXT2IMG + FaceDetailer 안면 복원 배전반</div>
          </div>
          <div class="wf-card" onclick="selectPresetWorkflow('z_image_turbo', this)">
            <div class="wf-card-header">
              <span class="wf-card-title">Z-Image Turbo 6B (Next-Gen DiT)</span>
              <span class="wf-card-badge">7 Nodes</span>
            </div>
            <div class="wf-card-desc">차세대 6B 파라미터 Diffusion Transformer 플로우 매칭 실사 고화질 배전반</div>
          </div>
          <div class="wf-card" onclick="selectPresetWorkflow('sdxs_lightning', this)">
            <div class="wf-card-header">
              <span class="wf-card-title">SDXS Lightning (High-Speed Mobile)</span>
              <span class="wf-card-badge">7 Nodes</span>
            </div>
            <div class="wf-card-desc">모바일 초고속 2스텝 경량화 디스틸레이션 실시간 추론 배전반</div>
          </div>
          <div class="wf-card" onclick="selectPresetWorkflow('anime_masterpiece', this)">
            <div class="wf-card-header">
              <span class="wf-card-title">Anime Masterpiece (DreamShaper8 LCM)</span>
              <span class="wf-card-badge">7 Nodes</span>
            </div>
            <div class="wf-card-desc">모바일 특화 6스텝 LCM 애니메 / 일러스트레이션 고화질 배전반</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Custom Workflow Upload</div>
        <div class="drop-zone" onclick="document.getElementById('wfFileInput').click()">
          <span style="font-weight: 600;">Load Workflow Specification (.json)</span>
          <span style="font-size: 12px; color: var(--text-muted);">ComfyUI UI Graph &amp; API Prompt Specification</span>
          <input type="file" id="wfFileInput" accept=".json" style="display: none;" onchange="handleWorkflowFile(this)">
        </div>
      </div>

      <!-- Interactive DAG Visualization & Inspector Section -->
      <div class="card" id="wfInfo" style="display: none;">
        <div class="card-title">
          <span>Workflow DAG Pipeline (<span id="wfStepCount">0</span> Nodes)</span>
        </div>

        <!-- Action Toolbar for Export and Save -->
        <div style="display: flex; gap: 8px; margin-bottom: 12px;">
          <button type="button" class="action-btn" style="flex: 1; padding: 10px 8px; font-size: 12px; font-weight: 700; background: #182238; border: 1px solid var(--accent); color: #38bdf8;" onclick="exportWorkflowJson()">
            <span>📥 Export JSON (추출)</span>
          </button>
          <button type="button" class="action-btn" style="flex: 1; padding: 10px 8px; font-size: 12px; font-weight: 700; background: #182238; border: 1px solid var(--accent); color: #a78bfa;" onclick="openSavePresetModal()">
            <span>💾 Save Preset (단말 저장)</span>
          </button>
        </div>

        <div class="dag-view-switcher">
          <button class="dag-view-btn active" id="btnViewFlow" onclick="switchDagView('flow')">Visual Flow</button>
          <button class="dag-view-btn" id="btnViewMatrix" onclick="switchDagView('matrix')">Topology Matrix</button>
          <button class="dag-view-btn" id="btnViewList" onclick="switchDagView('list')">Sequence List</button>
        </div>

        <!-- View 1: Visual Connected DAG Flow -->
        <div id="dagFlowView" class="dag-flow-container"></div>

        <!-- View 2: Multi-Lane Parallel Topology Matrix View -->
        <div id="dagMatrixView" class="dag-matrix-container" style="display: none;"></div>

        <!-- View 3: Compact Ordered List -->
        <div id="dagListView" class="node-sequence" style="display: none;"></div>
      </div>
    </div>

    <!-- Save Preset Modal -->
    <div id="savePresetModal" class="modal-overlay" style="display: none;">
      <div class="modal-card">
        <div class="modal-title">💾 Save Custom Workflow Preset</div>
        <div style="font-size: 12px; color: var(--text-muted);">현재 튜닝된 배전반 파라미터를 단말기 로컬 프리셋으로 저장합니다.</div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <label style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">프리셋 명칭 (Preset Name)</label>
          <input type="text" id="savePresetNameInput" class="custom-input" placeholder="e.g. My Photoreal 8K Tuned" value="">
          <label style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">고유 식별자 (Preset ID)</label>
          <input type="text" id="savePresetIdInput" class="custom-input" placeholder="e.g. my_photoreal_8k" value="">
        </div>
        <div style="display: flex; gap: 8px; margin-top: 10px;">
          <button type="button" class="action-btn" style="flex: 1; background: var(--accent); color: white; font-weight: 700;" onclick="confirmSavePreset()">저장 (Save)</button>
          <button type="button" class="action-btn" style="flex: 1;" onclick="closeSavePresetModal()">취소 (Cancel)</button>
        </div>
      </div>
    </div>

    <!-- Real-time Progress Display -->
    <div class="progress-box" id="progressBox">
      <div class="metrics-row">
        <span id="phaseText">Standby</span>
        <span id="stepCounter">0 / 0</span>
      </div>
      <div class="metrics-row" id="activeNodeRow" style="display: none; margin-top: 4px; font-size: 11px;">
        <span id="activeNodeBadge" style="background: rgba(99, 102, 241, 0.25); color: #a5b4fc; padding: 2px 6px; border-radius: 4px; font-weight: 700;">Node --</span>
        <span id="activeNodeTitle" style="color: #cbd5e1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"></span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="barFill"></div>
      </div>
      <div class="metrics-row">
        <span id="speedEtaText">Initializing...</span>
        <span id="liveElapsedEtaText" style="color: #38bdf8; font-weight: 700;">0.0s</span>
      </div>
    </div>

    <!-- Image Preview Result -->
    <div class="preview-box" id="previewBox">
      <img id="resultImg" class="preview-img" alt="Generated Image">
      <div class="metrics-row" style="width: 100%;">
        <span id="elapsedText">Elapsed: --</span>
        <span style="color: var(--success); font-weight: 700;">Complete (Exit 0)</span>
      </div>
      <div class="preview-actions">
        <a id="downloadLink" class="action-btn" download="generated_image.png" target="_blank">Download</a>
        <button class="action-btn" onclick="shareToGallery()">Gallery</button>
      </div>
    </div>
  </main>

  <!-- Pipeline Queue Modal -->
  <div id="queueModal" class="modal-overlay" style="display: none;">
    <div class="modal-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="modal-title">📋 Task Queue Pipeline (작업 대기열)</div>
        <button type="button" class="sheet-btn close" onclick="closeQueueModal()">✕</button>
      </div>
      <div style="font-size: 12px; color: var(--text-muted);">실행 중인 작업 완료 후 순차적으로 자동 실행됩니다.</div>
      <div id="queueListContainer" style="display: flex; flex-direction: column; gap: 6px; max-height: 240px; overflow-y: auto; margin-top: 6px;">
        <div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 12px;">대기열이 비어 있습니다.</div>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 10px;">
        <button type="button" class="action-btn" style="flex: 1; background: #b91c1c; color: white; font-weight: 700;" onclick="clearQueue()">대기열 비우기</button>
        <button type="button" class="action-btn" style="flex: 1;" onclick="closeQueueModal()">닫기</button>
      </div>
    </div>
  </div>

  <!-- Floating Draggable Bottom Sheet Overlay -->
  <div class="sheet-overlay" id="sheetOverlay" onclick="setSheetState('closed')"></div>

  <!-- Floating Draggable Bottom Sheet (화면 따라다니는 모바일 인스펙터) -->
  <div class="bottom-sheet" id="nodeInspectorSheet">
    <div class="sheet-handle-area" id="sheetDragHandle" onclick="cycleSheetState()">
      <div class="sheet-drag-pill"></div>
    </div>
    <div class="sheet-header-bar">
      <div class="sheet-header-left">
        <span class="flow-node-badge" id="inspClassBadge">CLASS</span>
        <div style="min-width:0; overflow:hidden;">
          <div style="font-size: 10px; font-weight: 700; color: var(--accent); text-transform: uppercase;" id="inspStage">Step 1 of 8</div>
          <div style="font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" id="inspTitle">Node Title</div>
        </div>
      </div>
      <div class="sheet-header-controls">
        <button type="button" class="sheet-btn" title="Peek (최소화)" onclick="setSheetState('peek')">─</button>
        <button type="button" class="sheet-btn" title="Half (중간)" onclick="setSheetState('half')">◫</button>
        <button type="button" class="sheet-btn" title="Full (최대화)" onclick="setSheetState('full')">⤢</button>
        <button type="button" class="sheet-btn close" title="Close (닫기)" onclick="setSheetState('closed')">✕</button>
      </div>
    </div>
    <div class="sheet-content">
      <div>
        <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">Operation Role</div>
        <div class="inspector-role" id="inspRole">Role description</div>
      </div>

      <div>
        <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">Configured Parameters (수정 가능)</div>
        <div id="inspPromptBox" class="inspector-prompt-box" style="display: none; margin-bottom: 8px;"></div>
        <div class="inspector-grid" id="inspParamsGrid"></div>
        <div style="display: flex; gap: 8px; margin-top: 10px;">
          <button type="button" class="action-btn" style="flex: 2; padding: 9px 12px; font-size: 12px; background: var(--accent); color: white; border-color: var(--accent); font-weight: 700;" onclick="applyNodeChanges()">
            <span>✓ Apply Changes (노드 수정 적용)</span>
          </button>
          <button type="button" class="action-btn" style="flex: 1; padding: 9px 12px; font-size: 12px;" onclick="resetNodeChanges()">
            <span>↺ Reset</span>
          </button>
        </div>
        <div id="nodeFeedbackMsg" style="font-size: 11px; color: var(--success); font-weight: 600; display: none; margin-top: 6px; padding: 6px 10px; background: rgba(16, 185, 129, 0.12); border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3);"></div>
      </div>

      <div>
        <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">Data Flow &amp; Wiring</div>
        <div class="wiring-box" id="inspWiringBox"></div>
      </div>
    </div>
  </div>

  <div class="bottom-bar">
    <button class="generate-btn" id="generateBtn" onclick="onGenerateClick()">
      <span>Generate</span>
    </button>
  </div>

  <script>
    let activeTab = 'quick';
    let currentModel = 'realistic';
    let loadedWorkflowData = null;
    let pollInterval = null;

    function switchTab(tab) {
      activeTab = tab;
      document.getElementById('tabQuickBtn').classList.toggle('active', tab === 'quick');
      document.getElementById('tabWfBtn').classList.toggle('active', tab === 'wf');
      document.getElementById('quickTab').style.display = tab === 'quick' ? 'flex' : 'none';
      document.getElementById('wfTab').style.display = tab === 'wf' ? 'flex' : 'none';
      
      const genBtn = document.getElementById('generateBtn');
      if (tab === 'quick') {
        genBtn.innerHTML = '<span>Generate</span>';
      } else {
        genBtn.innerHTML = '<span>Execute AmfyUI Workflow</span>';
      }
    }

    function selectModel(elem) {
      document.querySelectorAll('#modelChips .chip').forEach(c => c.classList.remove('active'));
      elem.classList.add('active');
      currentModel = elem.dataset.model;
      if (currentModel === 'z-image-turbo') {
        document.getElementById('stepsInput').value = 8;
        document.getElementById('stepsVal').innerText = 8;
        document.getElementById('cfgInput').value = 1.0;
        document.getElementById('cfgVal').innerText = 1.0;
      } else if (currentModel === 'sdxs') {
        document.getElementById('stepsInput').value = 2;
        document.getElementById('stepsVal').innerText = 2;
        document.getElementById('cfgInput').value = 1.0;
        document.getElementById('cfgVal').innerText = 1.0;
      } else if (currentModel === 'anime') {
        document.getElementById('stepsInput').value = 4;
        document.getElementById('stepsVal').innerText = 4;
        document.getElementById('cfgInput').value = 1.0;
        document.getElementById('cfgVal').innerText = 1.0;
        const pInput = document.getElementById('promptInput');
        if (!pInput.value || pInput.value.includes('professional portrait of an East Asian woman')) {
          pInput.value = '1girl, cute anime style, masterpiece, best quality, beautiful detailed eyes, vibrant colors';
        }
      } else {
        document.getElementById('stepsInput').value = 15;
        document.getElementById('stepsVal').innerText = 15;
        document.getElementById('cfgInput').value = 7.0;
        document.getElementById('cfgVal').innerText = 7.0;
      }
    }

    function addTag(text) {
      const p = document.getElementById('promptInput');
      p.value = p.value.trim() + ', ' + text;
    }

    let currentPresetId = 'standard_photoreal';

    async function loadPresetList() {
      try {
        const resp = await fetch('/api/workflows');
        if (!resp.ok) return;
        const data = await resp.json();
        const container = document.getElementById('wfPresetList');
        if (!container || !data.workflows) return;

        container.innerHTML = '';
        data.workflows.forEach(w => {
          const card = document.createElement('div');
          card.className = 'wf-card' + (w.id === currentPresetId ? ' active' : '');
          card.id = 'wfCard_' + w.id;
          card.onclick = () => selectPresetWorkflow(w.id, card);

          const badgeHtml = w.custom
            ? `<span class="wf-card-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">CUSTOM (${w.node_count}N)</span>`
            : `<span class="wf-card-badge">${w.node_count} Nodes</span>`;

          card.innerHTML = `
            <div class="wf-card-header">
              <span class="wf-card-title">${escapeHtml(w.name)}</span>
              ${badgeHtml}
            </div>
            <div class="wf-card-desc">${escapeHtml(w.description || '')}</div>
          `;
          container.appendChild(card);
        });
      } catch (err) {
        console.error('Failed fetching presets:', err);
      }
    }

    let deviceModelCatalog = { checkpoints: [], vae: [], loras: [], text_encoders: [], all: [] };

    async function loadDeviceModelCatalog() {
      try {
        const resp = await fetch('/api/models');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.model_catalog) {
          deviceModelCatalog = data.model_catalog;
        }
      } catch (e) {
        console.warn('Failed loading device model catalog:', e);
      }
    }

    function syncSlider(numId, val) {
      const el = document.getElementById(numId);
      if (el) el.value = val;
    }

    function syncNumber(sliderId, val) {
      const el = document.getElementById(sliderId);
      if (el) el.value = val;
    }

    function onModelSelectChange(key, selEl) {
      const customWrap = document.getElementById('customPathWrap_' + key);
      if (!customWrap) return;
      if (selEl.value === '__custom__') {
        customWrap.style.display = 'block';
        const inp = document.getElementById('editProp_' + key + '_custom');
        if (inp) inp.focus();
      } else {
        customWrap.style.display = 'none';
      }
    }

    function onDimSelectChange(key, selEl) {
      const customWrap = document.getElementById('customDimWrap_' + key);
      if (!customWrap) return;
      if (selEl.value === '__custom__') {
        customWrap.style.display = 'block';
        const inp = document.getElementById('editProp_' + key + '_custom');
        if (inp) inp.focus();
      } else {
        customWrap.style.display = 'none';
      }
    }

    let currentSheetState = 'closed';

    function setSheetState(state) {
      currentSheetState = state;
      const sheet = document.getElementById('nodeInspectorSheet');
      const overlay = document.getElementById('sheetOverlay');
      if (!sheet) return;
      sheet.classList.remove('peek', 'half', 'full', 'open');
      if (state !== 'closed') {
        sheet.classList.add('open', state);
      }
      if (overlay) {
        overlay.classList.toggle('active', state === 'full');
      }
    }

    function cycleSheetState() {
      if (currentSheetState === 'closed' || currentSheetState === 'peek') {
        setSheetState('half');
      } else if (currentSheetState === 'half') {
        setSheetState('full');
      } else {
        setSheetState('peek');
      }
    }

    function setupBottomSheetDrag() {
      const handle = document.getElementById('sheetDragHandle');
      if (!handle) return;
      let startY = 0;
      let startHeight = 0;
      let isDragging = false;
      const sheet = document.getElementById('nodeInspectorSheet');

      const onStart = (clientY) => {
        isDragging = true;
        startY = clientY;
        startHeight = sheet.offsetHeight;
        sheet.style.transition = 'none';
      };

      const onMove = (clientY) => {
        if (!isDragging) return;
        const deltaY = startY - clientY;
        const newHeight = Math.max(60, Math.min(window.innerHeight * 0.88, startHeight + deltaY));
        sheet.style.height = newHeight + 'px';
      };

      const onEnd = () => {
        if (!isDragging) return;
        isDragging = false;
        sheet.style.transition = '';
        const currentH = sheet.offsetHeight;
        sheet.style.height = '';
        const vh = window.innerHeight;
        if (currentH < 110) {
          setSheetState('peek');
        } else if (currentH < vh * 0.65) {
          setSheetState('half');
        } else {
          setSheetState('full');
        }
      };

      handle.addEventListener('touchstart', e => {
        if (e.touches.length === 1) onStart(e.touches[0].clientY);
      }, { passive: true });
      window.addEventListener('touchmove', e => {
        if (isDragging && e.touches.length === 1) onMove(e.touches[0].clientY);
      }, { passive: true });
      window.addEventListener('touchend', () => { onEnd(); });

      handle.addEventListener('mousedown', e => {
        onStart(e.clientY);
        const moveHandler = ev => onMove(ev.clientY);
        const upHandler = () => {
          onEnd();
          window.removeEventListener('mousemove', moveHandler);
          window.removeEventListener('mouseup', upHandler);
        };
        window.addEventListener('mousemove', moveHandler);
        window.addEventListener('mouseup', upHandler);
      });
    }

    async function selectPresetWorkflow(wfId, elem, autoScroll = true) {
      currentPresetId = wfId;
      try {
        localStorage.setItem('amfy_selected_workflow', wfId);
        localStorage.setItem('amvi_selected_workflow', wfId);
      } catch (e) {}
      document.querySelectorAll('.wf-card').forEach(c => c.classList.remove('active'));
      if (elem) {
        elem.classList.add('active');
      } else {
        const target = document.getElementById('wfCard_' + wfId);
        if (target) target.classList.add('active');
      }
      try {
        const resp = await fetch('/api/workflows/' + wfId);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        loadedWorkflowData = data;
        await inspectAndRenderWorkflow(data);

        // 1. Smooth auto-scroll down to DAG Pipeline if requested
        if (autoScroll) {
          const wfInfoEl = document.getElementById('wfInfo');
          if (wfInfoEl) {
            wfInfoEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
      } catch (err) {
        console.error('Failed loading workflow:', err);
      }
    }

    let currentInspectionData = null;
    let selectedNodeId = null;
    let currentDagView = 'flow';

    function switchDagView(mode) {
      currentDagView = mode;
      document.getElementById('btnViewFlow').classList.toggle('active', mode === 'flow');
      document.getElementById('btnViewMatrix').classList.toggle('active', mode === 'matrix');
      document.getElementById('btnViewList').classList.toggle('active', mode === 'list');
      document.getElementById('dagFlowView').style.display = mode === 'flow' ? 'flex' : 'none';
      document.getElementById('dagMatrixView').style.display = mode === 'matrix' ? 'flex' : 'none';
      document.getElementById('dagListView').style.display = mode === 'list' ? 'flex' : 'none';
      if (mode === 'matrix' && currentInspectionData) {
        renderTopologyMatrix(currentInspectionData);
      }
    }

    function getNodeCategory(classType) {
      const c = (classType || '').toLowerCase();
      if (c.includes('loader') || c.includes('checkpoint')) return { cls: 'loader', label: 'LOADER' };
      if (c.includes('clip') || c.includes('text') || c.includes('conditioning')) return { cls: 'conditioning', label: 'PROMPT' };
      if (c.includes('sampler')) return { cls: 'sampler', label: 'SAMPLER' };
      if (c.includes('vae')) return { cls: 'decode', label: 'DECODER' };
      if (c.includes('detailer')) return { cls: 'detailer', label: 'DETAILER' };
      if (c.includes('save') || c.includes('preview')) return { cls: 'save', label: 'OUTPUT' };
      return { cls: '', label: classType };
    }

    function renderInspectionView(data) {
      document.getElementById('wfInfo').style.display = 'block';
      document.getElementById('wfStepCount').innerText = data.total_nodes;

      const flowContainer = document.getElementById('dagFlowView');
      flowContainer.innerHTML = '';

      const listContainer = document.getElementById('dagListView');
      listContainer.innerHTML = '';

      const totalSteps = data.execution_order.length;

      data.execution_order.forEach((step, idx) => {
        const nid = step.node_id;
        const nodeDetail = (data.nodes_detail && data.nodes_detail[nid]) || step;
        const cat = getNodeCategory(step.class_type);

        const flowCard = document.createElement('div');
        flowCard.className = 'flow-node-card';
        flowCard.id = 'flowNode_' + nid;
        flowCard.dataset.nodeId = nid;
        flowCard.onclick = () => inspectNode(nid);

        let paramChips = [];
        if (nodeDetail.parameters) {
          for (const [k, v] of Object.entries(nodeDetail.parameters)) {
            if (k !== 'text' && typeof v !== 'object') {
              let vStr = String(v);
              if (vStr.length > 20) vStr = vStr.substring(0, 18) + '..';
              paramChips.push(`<span class="flow-param-chip">${escapeHtml(k)}: ${escapeHtml(vStr)}</span>`);
            }
          }
        }

        // Multi-Branch IN ports
        let inChips = [];
        if (nodeDetail.connections_in && nodeDetail.connections_in.length > 0) {
          nodeDetail.connections_in.forEach(c => {
            inChips.push(`<span class="port-chip in">${escapeHtml(c.slot)} ← #${c.from_node}</span>`);
          });
        }

        // Multi-Branch OUT ports
        let outChips = [];
        if (nodeDetail.connections_out && nodeDetail.connections_out.length > 0) {
          const slotMap = {};
          nodeDetail.connections_out.forEach(c => {
            if (!slotMap[c.slot]) slotMap[c.slot] = [];
            slotMap[c.slot].push('#' + c.to_node);
          });
          for (const [slot, targets] of Object.entries(slotMap)) {
            outChips.push(`<span class="port-chip out">${escapeHtml(slot)} ➔ ${targets.join(', ')}</span>`);
          }
        }

        let portsHtml = '';
        if (inChips.length > 0) {
          portsHtml += `<div class="flow-ports-in"><span class="ports-label">IN (${nodeDetail.connections_in.length}):</span>${inChips.join('')}</div>`;
        }
        if (outChips.length > 0) {
          portsHtml += `<div class="flow-ports-out"><span class="ports-label">OUT (${nodeDetail.connections_out.length}):</span>${outChips.join('')}</div>`;
        }

        flowCard.innerHTML = `
          <div class="flow-node-header">
            <span class="flow-node-title">
              <span class="node-num">${step.step}</span>
              <span>${escapeHtml(step.title || step.class_type)}</span>
            </span>
            <span class="flow-node-badge ${cat.cls}">${cat.label}</span>
          </div>
          ${paramChips.length > 0 ? `<div class="flow-node-params">${paramChips.slice(0, 3).join('')}</div>` : ''}
          ${portsHtml}
        `;
        flowContainer.appendChild(flowCard);

        if (idx < totalSteps - 1) {
          const arrow = document.createElement('div');
          arrow.className = 'flow-arrow-down';
          arrow.innerText = '↓';
          flowContainer.appendChild(arrow);
        }

        const pill = document.createElement('div');
        pill.className = 'node-pill';
        pill.id = 'listNode_' + nid;
        pill.style.cursor = 'pointer';
        pill.onclick = () => inspectNode(nid);
        pill.innerHTML = `
          <span class="node-num">${step.step}</span>
          <span style="flex:1;">[#${nid}] ${escapeHtml(step.title || step.class_type)}</span>
          <span class="flow-node-badge ${cat.cls}">${cat.label}</span>
        `;
        listContainer.appendChild(pill);
      });

      renderTopologyMatrix(data);
    }

    function renderTopologyMatrix(data) {
      const matrixContainer = document.getElementById('dagMatrixView');
      if (!matrixContainer) return;
      matrixContainer.innerHTML = '';

      const layers = [
        { id: 'gen', title: 'Layer 1: Input & Generators (입력 및 생성)', nodes: [] },
        { id: 'cond', title: 'Layer 2: Conditioning & Prompts (병렬 분기)', nodes: [], isParallel: true },
        { id: 'sampler', title: 'Layer 3: Latent Diffusion Core (확산 코어 수렴)', nodes: [] },
        { id: 'decode', title: 'Layer 4: Latent Space Decode (잠재 복원)', nodes: [] },
        { id: 'output', title: 'Layer 5: Refinement & Image Output (보정 및 최종 출력)', nodes: [] }
      ];

      data.execution_order.forEach(step => {
        const c = (step.class_type || '').toLowerCase();
        if (c.includes('loader') || c.includes('checkpoint') || c.includes('latent') || c.includes('lora')) {
          layers[0].nodes.push(step);
        } else if (c.includes('clip') || c.includes('conditioning') || c.includes('text')) {
          layers[1].nodes.push(step);
        } else if (c.includes('sampler')) {
          layers[2].nodes.push(step);
        } else if (c.includes('vae')) {
          layers[3].nodes.push(step);
        } else {
          layers[4].nodes.push(step);
        }
      });

      layers.forEach(layer => {
        if (layer.nodes.length === 0) return;
        const layerCard = document.createElement('div');
        layerCard.className = 'matrix-layer';

        let convergenceHtml = '';
        if (layer.id === 'sampler') {
          convergenceHtml = '<div class="matrix-convergence-chip">⛙ Converges 4 Inputs: MODEL + POS + NEG + LATENT</div>';
        } else if (layer.id === 'decode') {
          convergenceHtml = '<div class="matrix-convergence-chip">⛙ Converges 2 Inputs: LATENT_SAMPLES + VAE</div>';
        }

        layerCard.innerHTML = `
          <div class="matrix-layer-title">
            <span>${layer.title}</span>
            <span>${layer.nodes.length} Nodes</span>
          </div>
          ${convergenceHtml}
        `;

        const row = document.createElement('div');
        row.className = layer.isParallel ? 'matrix-row-parallel' : 'dag-flow-container';
        row.style.gap = '6px';

        layer.nodes.forEach(step => {
          const nid = step.node_id;
          const nodeDetail = (data.nodes_detail && data.nodes_detail[nid]) || step;
          const cat = getNodeCategory(step.class_type);

          const card = document.createElement('div');
          card.className = 'flow-node-card';
          card.id = 'matrixNode_' + nid;
          card.dataset.nodeId = nid;
          card.onclick = () => inspectNode(nid);

          let inCount = (nodeDetail.connections_in || []).length;
          let outCount = (nodeDetail.connections_out || []).length;

          card.innerHTML = `
            <div class="flow-node-header">
              <span class="flow-node-title">
                <span class="node-num">${step.step}</span>
                <span style="font-size:12px;">${escapeHtml(step.title || step.class_type)}</span>
              </span>
              <span class="flow-node-badge ${cat.cls}">${cat.label}</span>
            </div>
            <div style="display:flex;gap:6px;font-size:10px;color:var(--text-muted);margin-top:2px;">
              <span>IN: ${inCount} slots</span> | <span>OUT: ${outCount} slots</span>
            </div>
          `;
          row.appendChild(card);
        });

        layerCard.appendChild(row);
        matrixContainer.appendChild(layerCard);
      });
    }

    async function inspectAndRenderWorkflow(wfJson) {
      try {
        const resp = await fetch('/api/workflow/inspect', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(wfJson)
        });
        const data = await resp.json();
        if (data.error) {
          alert('Workflow Parse Error: ' + data.error);
          return;
        }
        currentInspectionData = data;
        renderInspectionView(data);

        // Auto-inspect first node or KSampler
        const samplerStep = data.execution_order.find(s => s.class_type.toLowerCase().includes('sampler'));
        const targetNodeId = samplerStep ? samplerStep.node_id : data.execution_order[0].node_id;
        inspectNode(targetNodeId);

      } catch (e) {
        alert('Workflow Inspection Failed: ' + e.message);
      }
    }

    function inspectNode(nodeId) {
      if (!currentInspectionData || !currentInspectionData.nodes_detail) return;
      selectedNodeId = nodeId;
      const node = currentInspectionData.nodes_detail[nodeId];
      if (!node) return;

      // Multi-Branch Upstream & Downstream Resolution
      const sourceNodeIds = new Set();
      if (node.connections_in) {
        node.connections_in.forEach(c => sourceNodeIds.add(String(c.from_node)));
      }
      const targetNodeIds = new Set();
      if (node.connections_out) {
        node.connections_out.forEach(c => targetNodeIds.add(String(c.to_node)));
      }

      // Apply branch illumination and dimming to all flow and matrix cards
      currentInspectionData.execution_order.forEach(step => {
        const sNid = String(step.node_id);
        const flowCard = document.getElementById('flowNode_' + sNid);
        const matrixCard = document.getElementById('matrixNode_' + sNid);

        [flowCard, matrixCard].forEach(card => {
          if (!card) return;
          card.classList.remove('selected', 'flow-node-source', 'flow-node-target', 'flow-node-dimmed');
          if (sNid === String(nodeId)) {
            card.classList.add('selected');
          } else if (sourceNodeIds.has(sNid)) {
            card.classList.add('flow-node-source');
          } else if (targetNodeIds.has(sNid)) {
            card.classList.add('flow-node-target');
          } else {
            card.classList.add('flow-node-dimmed');
          }
        });
      });

      document.querySelectorAll('.node-pill').forEach(p => p.style.borderColor = 'var(--card-border)');
      const activePill = document.getElementById('listNode_' + nodeId);
      if (activePill) activePill.style.borderColor = 'var(--accent)';

      // Open floating bottom sheet in half peek if closed
      if (currentSheetState === 'closed') {
        setSheetState('half');
      }

      const cat = getNodeCategory(node.class_type);
      document.getElementById('inspStage').innerText = `Step ${node.step} of ${currentInspectionData.total_nodes}`;
      document.getElementById('inspTitle').innerText = `[#${node.node_id}] ${node.title || node.class_type}`;
      const badge = document.getElementById('inspClassBadge');
      badge.className = 'flow-node-badge ' + cat.cls;
      badge.innerText = cat.label;

      document.getElementById('inspRole').innerText = node.role || 'Executes this pipeline transformation stage.';

      // Parameters
      const promptBox = document.getElementById('inspPromptBox');
      const paramsGrid = document.getElementById('inspParamsGrid');
      paramsGrid.innerHTML = '';
      const feedback = document.getElementById('nodeFeedbackMsg');
      if (feedback) feedback.style.display = 'none';

      if (node.parameters && node.parameters.text !== undefined) {
        promptBox.style.display = 'block';
        promptBox.innerHTML = `
          <div style="font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 4px;">PROMPT TEXT (수정 가능):</div>
          <textarea id="editProp_text" class="custom-textarea" rows="3">${escapeHtml(String(node.parameters.text))}</textarea>
        `;
      } else {
        promptBox.style.display = 'none';
      }

      let hasProps = false;
      if (node.parameters) {
        for (const [k, v] of Object.entries(node.parameters)) {
          if (k === 'text') continue;
          hasProps = true;
          const prop = document.createElement('div');
          const kLow = k.toLowerCase();

          let inputHtml = '';

          // 1. Checkpoint / Diffusion Model Selector
          if (['ckpt_name', 'model_name', 'checkpoint', 'unet_name'].includes(kLow)) {
            prop.className = 'inspector-prop full-span';
            let ckptOptions = [];
            const availableCkpts = (deviceModelCatalog.checkpoints && deviceModelCatalog.checkpoints.length > 0)
              ? deviceModelCatalog.checkpoints
              : [
                  { name: 'realistic.gguf', size_mb: 1546.8 },
                  { name: 'z_image_turbo-Q4_0.gguf', size_mb: 3512.7 },
                  { name: 'z_image_turbo-Q2_K.gguf', size_mb: 2472.3 },
                  { name: 'sdxs.gguf', size_mb: 651.2 },
                  { name: 'anime.gguf', size_mb: 1550.0 },
                  { name: 'counterfeit-v3-lcm.gguf', size_mb: 1494.2 },
                  { name: 'ghostmix.Q4_K_M.gguf', size_mb: 461.8 },
                  { name: 'anything-v5.q4_0.gguf', size_mb: 1497.7 },
                  { name: 'anima-preview-Q4_K.gguf', size_mb: 1147.4 }
                ];

            let matched = false;
            availableCkpts.forEach(m => {
              const isSel = (String(v) === m.name);
              if (isSel) matched = true;
              ckptOptions.push(`<option value="${escapeHtml(m.name)}" ${isSel ? 'selected' : ''}>${escapeHtml(m.name)} (${m.size_mb} MB)</option>`);
            });

            if (!matched && v) {
              ckptOptions.unshift(`<option value="${escapeHtml(String(v))}" selected>${escapeHtml(String(v))} (Preset Default)</option>`);
            }
            ckptOptions.push(`<option value="__custom__">✏️ Direct File Path / Custom...</option>`);

            inputHtml = `
              <select id="editProp_${k}" class="custom-select" onchange="onModelSelectChange('${k}', this)">
                ${ckptOptions.join('')}
              </select>
              <div id="customPathWrap_${k}" class="custom-path-box">
                <input type="text" id="editProp_${k}_custom" class="custom-input" placeholder="e.g. realistic.gguf or absolute path" value="${escapeHtml(String(v))}">
              </div>
            `;
            prop.innerHTML = `
              <span class="inspector-prop-key">${escapeHtml(k)} <span class="model-badge-tag">DEVICE MODELS (${availableCkpts.length})</span></span>
              ${inputHtml}
            `;
            paramsGrid.appendChild(prop);
            continue;
          }

          // 2. VAE Model Selector
          if (['vae_name', 'vae'].includes(kLow)) {
            prop.className = 'inspector-prop full-span';
            let vaeOptions = [
              `<option value="Baked-in" ${String(v) === 'Baked-in' ? 'selected' : ''}>Baked-in (내장 VAE)</option>`
            ];
            const availableVae = (deviceModelCatalog.vae && deviceModelCatalog.vae.length > 0)
              ? deviceModelCatalog.vae
              : [
                  { name: 'taef1.safetensors', size_mb: 9.4 },
                  { name: 'z_image_ae.safetensors', size_mb: 319.8 }
                ];

            let matched = (String(v) === 'Baked-in');
            availableVae.forEach(m => {
              const isSel = (String(v) === m.name);
              if (isSel) matched = true;
              vaeOptions.push(`<option value="${escapeHtml(m.name)}" ${isSel ? 'selected' : ''}>${escapeHtml(m.name)} (${m.size_mb} MB)</option>`);
            });

            if (!matched && v) {
              vaeOptions.unshift(`<option value="${escapeHtml(String(v))}" selected>${escapeHtml(String(v))} (Preset Default)</option>`);
            }
            vaeOptions.push(`<option value="__custom__">✏️ Direct File Path / Custom...</option>`);

            inputHtml = `
              <select id="editProp_${k}" class="custom-select" onchange="onModelSelectChange('${k}', this)">
                ${vaeOptions.join('')}
              </select>
              <div id="customPathWrap_${k}" class="custom-path-box">
                <input type="text" id="editProp_${k}_custom" class="custom-input" placeholder="e.g. taef1.safetensors" value="${escapeHtml(String(v))}">
              </div>
            `;
            prop.innerHTML = `
              <span class="inspector-prop-key">${escapeHtml(k)} <span class="model-badge-tag">VAE WEIGHTS</span></span>
              ${inputHtml}
            `;
            paramsGrid.appendChild(prop);
            continue;
          }

          // 3. LoRA Model Selector
          if (['lora_name', 'lora'].includes(kLow)) {
            prop.className = 'inspector-prop full-span';
            let loraOptions = [
              `<option value="None" ${String(v) === 'None' ? 'selected' : ''}>None (미사용)</option>`
            ];
            const availableLoras = deviceModelCatalog.loras || [];
            let matched = (String(v) === 'None');
            availableLoras.forEach(m => {
              const isSel = (String(v) === m.name);
              if (isSel) matched = true;
              loraOptions.push(`<option value="${escapeHtml(m.name)}" ${isSel ? 'selected' : ''}>${escapeHtml(m.name)} (${m.size_mb} MB)</option>`);
            });
            if (!matched && v) {
              loraOptions.unshift(`<option value="${escapeHtml(String(v))}" selected>${escapeHtml(String(v))}</option>`);
            }
            loraOptions.push(`<option value="__custom__">✏️ Direct File Path / Custom...</option>`);

            inputHtml = `
              <select id="editProp_${k}" class="custom-select" onchange="onModelSelectChange('${k}', this)">
                ${loraOptions.join('')}
              </select>
              <div id="customPathWrap_${k}" class="custom-path-box">
                <input type="text" id="editProp_${k}_custom" class="custom-input" placeholder="e.g. detail_tweaker.safetensors" value="${escapeHtml(String(v))}">
              </div>
            `;
            prop.innerHTML = `
              <span class="inspector-prop-key">${escapeHtml(k)} <span class="model-badge-tag">LORA</span></span>
              ${inputHtml}
            `;
            paramsGrid.appendChild(prop);
            continue;
          }

          // 4. Sliders with Live Synchronized Number Inputs
          prop.className = 'inspector-prop';
          if (k === 'steps') {
            inputHtml = `
              <div class="param-slider-wrap">
                <input type="range" id="editProp_steps_slider" min="1" max="100" value="${v}" oninput="syncSlider('editProp_steps', this.value)">
                <input type="number" id="editProp_steps" class="custom-input param-num-compact" min="1" max="100" value="${v}" oninput="syncNumber('editProp_steps_slider', this.value)">
              </div>`;
          } else if (k === 'cfg') {
            inputHtml = `
              <div class="param-slider-wrap">
                <input type="range" id="editProp_cfg_slider" min="1.0" max="30.0" step="0.5" value="${v}" oninput="syncSlider('editProp_cfg', this.value)">
                <input type="number" id="editProp_cfg" class="custom-input param-num-compact" min="1.0" max="30.0" step="0.5" value="${v}" oninput="syncNumber('editProp_cfg_slider', this.value)">
              </div>`;
          } else if (k === 'denoise') {
            inputHtml = `
              <div class="param-slider-wrap">
                <input type="range" id="editProp_denoise_slider" min="0.0" max="1.0" step="0.05" value="${v}" oninput="syncSlider('editProp_denoise', this.value)">
                <input type="number" id="editProp_denoise" class="custom-input param-num-compact" min="0.0" max="1.0" step="0.05" value="${v}" oninput="syncNumber('editProp_denoise_slider', this.value)">
              </div>`;
          } else if (k === 'bbox_threshold') {
            inputHtml = `
              <div class="param-slider-wrap">
                <input type="range" id="editProp_bbox_threshold_slider" min="0.1" max="0.95" step="0.05" value="${v}" oninput="syncSlider('editProp_bbox_threshold', this.value)">
                <input type="number" id="editProp_bbox_threshold" class="custom-input param-num-compact" min="0.1" max="0.95" step="0.05" value="${v}" oninput="syncNumber('editProp_bbox_threshold_slider', this.value)">
              </div>`;
          } else if (k === 'seed') {
            inputHtml = `
              <div style="display:flex;gap:4px;align-items:center;">
                <input type="number" id="editProp_seed" class="custom-input" style="flex:1;" value="${v}">
                <button type="button" class="action-btn" style="padding:4px 8px;font-size:11px;" title="난수 시드 발생" onclick="randomizeInspectorSeed()">🎲</button>
              </div>`;
          } else if (k === 'sampler_name') {
            const samplers = ['euler_a', 'euler', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_sde', 'dpmpp_2s_ancestral', 'heun', 'lms', 'ddim', 'uni_pc', 'flow_euler'];
            const opts = samplers.map(s => `<option value="${s}" ${s === v ? 'selected' : ''}>${s}</option>`).join('');
            inputHtml = `<select id="editProp_sampler_name" class="custom-select">${opts}</select>`;
          } else if (k === 'scheduler') {
            const schedulers = ['karras', 'normal', 'exponential', 'simple', 'sgm_uniform', 'turbo', 'linear', 'ddim_uniform'];
            const opts = schedulers.map(s => `<option value="${s}" ${s === v ? 'selected' : ''}>${s}</option>`).join('');
            inputHtml = `<select id="editProp_scheduler" class="custom-select">${opts}</select>`;
          } else if (k === 'width' || k === 'height' || k === 'guide_size') {
            const dims = [256, 384, 512, 640, 768, 896, 1024, 1280];
            let matched = false;
            let opts = dims.map(d => {
              if (d == v) matched = true;
              return `<option value="${d}" ${d == v ? 'selected' : ''}>${d}px</option>`;
            }).join('');
            if (!matched && v) {
              opts = `<option value="${v}" selected>${v}px (Custom)</option>` + opts;
            }
            opts += `<option value="__custom__">✏️ Direct px...</option>`;
            inputHtml = `
              <select id="editProp_${k}" class="custom-select" onchange="onDimSelectChange('${k}', this)">${opts}</select>
              <div id="customDimWrap_${k}" class="custom-path-box">
                <input type="number" id="editProp_${k}_custom" class="custom-input" placeholder="e.g. 512" value="${v}">
              </div>
            `;
          } else if (typeof v === 'boolean' || ['true', 'false'].includes(String(v).toLowerCase())) {
            const bVal = (String(v).toLowerCase() === 'true');
            inputHtml = `
              <select id="editProp_${k}" class="custom-select">
                <option value="true" ${bVal ? 'selected' : ''}>True (활성화)</option>
                <option value="false" ${!bVal ? 'selected' : ''}>False (비활성화)</option>
              </select>`;
          } else if (typeof v === 'number') {
            inputHtml = `<input type="number" id="editProp_${k}" class="custom-input" value="${v}">`;
          } else {
            inputHtml = `<input type="text" id="editProp_${k}" class="custom-input" value="${escapeHtml(String(v))}">`;
          }

          prop.innerHTML = `
            <span class="inspector-prop-key">${escapeHtml(k)}</span>
            ${inputHtml}
          `;
          paramsGrid.appendChild(prop);
        }
      }
      if (!hasProps && promptBox.style.display === 'none') {
        paramsGrid.innerHTML = '<span style="font-size: 12px; color: var(--text-muted); grid-column: span 2;">Default model parameters applied</span>';
      }

      // Multi-Branch Wiring Explorer
      const wiringBox = document.getElementById('inspWiringBox');
      wiringBox.innerHTML = '';

      let hasWiring = false;
      if (node.connections_in && node.connections_in.length > 0) {
        hasWiring = true;
        node.connections_in.forEach(c => {
          const row = document.createElement('div');
          row.className = 'wiring-row';
          row.innerHTML = `<span class="slot-name">📥 ${escapeHtml(c.slot)}</span><span class="target-node">from [#${c.from_node}] ${escapeHtml(c.from_title || '')}</span>`;
          wiringBox.appendChild(row);
        });
      }

      if (node.connections_out && node.connections_out.length > 0) {
        hasWiring = true;
        node.connections_out.forEach(c => {
          const row = document.createElement('div');
          row.className = 'wiring-row';
          row.innerHTML = `<span class="slot-name">📤 ${escapeHtml(c.slot)}</span><span class="target-node">to [#${c.to_node}] ${escapeHtml(c.to_title || '')}</span>`;
          wiringBox.appendChild(row);
        });
      }

      if (!hasWiring) {
        wiringBox.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Root / Self-contained node</span>';
      }
    }

    function randomizeInspectorSeed() {
      const el = document.getElementById('editProp_seed');
      if (el) {
        const randSeed = Math.floor(Math.random() * 2147483647);
        el.value = randSeed;
      }
    }

    async function applyNodeChanges() {
      if (!loadedWorkflowData || !selectedNodeId || !currentInspectionData) return;
      const node = currentInspectionData.nodes_detail[selectedNodeId];
      if (!node) return;

      const newParams = {};
      if (node.parameters && node.parameters.text !== undefined) {
        const textEl = document.getElementById('editProp_text');
        if (textEl) newParams.text = textEl.value;
      }

      if (node.parameters) {
        for (const k of Object.keys(node.parameters)) {
          if (k === 'text') continue;
          const el = document.getElementById('editProp_' + k);
          if (el) {
            let val = el.value;
            // Handle custom path / dimension input
            if (val === '__custom__') {
              const customEl = document.getElementById('editProp_' + k + '_custom');
              if (customEl && customEl.value.trim() !== '') {
                val = customEl.value.trim();
              }
            }
            // Parse typed values
            if (typeof node.parameters[k] === 'boolean' || val === 'true' || val === 'false') {
              val = (String(val).toLowerCase() === 'true');
            } else if (['steps', 'seed', 'width', 'height', 'batch_size', 'guide_size'].includes(k)) {
              val = parseInt(val, 10);
            } else if (['cfg', 'denoise', 'bbox_threshold'].includes(k)) {
              val = parseFloat(val);
            } else if (typeof node.parameters[k] === 'number') {
              val = Number(val);
            }
            newParams[k] = val;
          }
        }
      }

      try {
        const resp = await fetch('/api/workflow/update_node', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            workflow: loadedWorkflowData,
            node_id: selectedNodeId,
            parameters: newParams
          })
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          throw new Error(data.error || 'Failed updating node');
        }

        loadedWorkflowData = data.workflow;
        currentInspectionData = data.inspection;

        renderInspectionView(data.inspection);
        inspectNode(selectedNodeId);

        const msg = document.getElementById('nodeFeedbackMsg');
        msg.innerText = `✓ Node #${selectedNodeId} (${node.title || node.class_type}) updated in active workflow.`;
        msg.style.display = 'block';
        setTimeout(() => { msg.style.display = 'none'; }, 3000);
      } catch (err) {
        alert('Failed to update node: ' + err.message);
      }
    }

    function resetNodeChanges() {
      if (selectedNodeId) inspectNode(selectedNodeId);
    }

    function exportWorkflowJson() {
      if (!loadedWorkflowData) {
        alert('No active workflow loaded to export.');
        return;
      }
      const wfName = (loadedWorkflowData.name || 'custom_workflow').toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      const dataStr = JSON.stringify(loadedWorkflowData, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = wfName + '.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    function openSavePresetModal() {
      if (!loadedWorkflowData) {
        alert('No active workflow loaded to save.');
        return;
      }
      const defName = loadedWorkflowData.name || 'Custom Workflow';
      const defId = defName.toLowerCase().replace(/[^a-z0-9_-]/g, '_') + '_tuned';
      document.getElementById('savePresetNameInput').value = defName + ' (Tuned)';
      document.getElementById('savePresetIdInput').value = defId;
      document.getElementById('savePresetModal').style.display = 'flex';
    }

    function closeSavePresetModal() {
      document.getElementById('savePresetModal').style.display = 'none';
    }

    async function confirmSavePreset() {
      const name = document.getElementById('savePresetNameInput').value.trim();
      const id = document.getElementById('savePresetIdInput').value.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      if (!name || !id) {
        alert('Please provide both Preset Name and Identifier.');
        return;
      }
      try {
        const resp = await fetch('/api/workflow/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            workflow: loadedWorkflowData,
            id: id,
            name: name,
            description: 'User tuned custom workflow on device'
          })
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          throw new Error(data.error || 'Failed saving preset');
        }
        closeSavePresetModal();
        alert('Saved as custom preset "' + name + '" on device!');
        await loadPresetList();
        selectPresetWorkflow(id);
      } catch (err) {
        alert('Error saving preset: ' + err.message);
      }
    }

    function escapeHtml(str) {
      return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function handleWorkflowFile(input) {
      const file = input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          loadedWorkflowData = JSON.parse(e.target.result);
          document.querySelectorAll('.wf-card').forEach(c => c.classList.remove('active'));
          await inspectAndRenderWorkflow(loadedWorkflowData);
        } catch (err) {
          alert('JSON Syntax Error: ' + err.message);
        }
      };
      reader.readAsText(file);
    }

    let isRunning = false;
    let clientTickerInterval = null;
    let backgroundHudInterval = null;
    let currentQueueSnapshot = { total: 0, items: [] };

    function startClientTicker(serverStartTime) {
      if (clientTickerInterval) clearInterval(clientTickerInterval);
      const baseStart = (serverStartTime && serverStartTime > 0) ? (serverStartTime * 1000) : Date.now();
      clientTickerInterval = setInterval(() => {
        if (!isRunning) {
          clearInterval(clientTickerInterval);
          clientTickerInterval = null;
          return;
        }
        const elapsedSec = Math.max(0, (Date.now() - baseStart) / 1000);
        const formatted = elapsedSec.toFixed(1) + 's';
        const hudTime = document.getElementById('hudTimeVal');
        if (hudTime) hudTime.innerText = formatted;
        const liveElapsed = document.getElementById('liveElapsedEtaText');
        if (liveElapsed) liveElapsed.innerText = 'Elapsed: ' + formatted;
      }, 100);
    }

    async function onGenerateClick() {
      const genBtn = document.getElementById('generateBtn');
      document.getElementById('previewBox').classList.remove('active');
      document.getElementById('progressBox').classList.add('active');

      let endpoint = '/api/generate';
      let payload = {};

      if (activeTab === 'quick') {
        endpoint = '/api/generate';
        const safeModeElem = document.getElementById('safeModeSwitch');
        const isSafe = safeModeElem ? safeModeElem.checked : true;
        payload = {
          prompt: document.getElementById('promptInput').value,
          model: currentModel,
          steps: parseInt(document.getElementById('stepsInput').value),
          cfg: parseFloat(document.getElementById('cfgInput').value),
          detail_faces: document.getElementById('detailerSwitch').checked,
          safe_mode: isSafe,
          gpu_duty_cycle: isSafe ? 90 : 100,
          device: 'gpu'
        };
      } else {
        if (!loadedWorkflowData) {
          alert('Please select an AmfyUI / ComfyUI workflow JSON file first.');
          document.getElementById('progressBox').classList.remove('active');
          return;
        }
        endpoint = '/api/workflow/run';
        payload = { workflow: loadedWorkflowData, device: 'gpu' };
      }

      try {
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const res = await resp.json();
        if (res.status === 'queued') {
          const qEl = document.getElementById('hudQueueVal');
          if (qEl) qEl.innerText = `${res.queue_total} pending`;
          const activeNodeTitle = document.getElementById('activeNodeTitle');
          if (activeNodeTitle) {
            activeNodeTitle.innerText = `+ Enqueued as Task #${res.queue_position} (${res.job_id})`;
            document.getElementById('activeNodeRow').style.display = 'flex';
          }
        }
      } catch (err) {
        console.error('Request dispatch failed:', err);
      }

      startProgressPolling();
    }

    function startProgressPolling() {
      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(async () => {
        try {
          const resp = await fetch('/api/progress');
          const data = await resp.json();
          updateUIWithProgress(data);
        } catch (e) {
          console.error(e);
        }
      }, 400);
    }

    function updateUIWithProgress(data) {
      const badge = document.getElementById('statusBadge');
      const statusText = document.getElementById('statusText');
      const genBtn = document.getElementById('generateBtn');

      // 1. Live Hardware Resource HUD Updates
      if (data.metrics) {
        const m = data.metrics;
        const ramEl = document.getElementById('hudRamVal');
        if (ramEl) {
          ramEl.innerText = `${m.ram_used_gb}/${m.ram_total_gb} GB (${m.ram_percent}%)`;
        }
        const cpuEl = document.getElementById('hudCpuVal');
        if (cpuEl) {
          cpuEl.innerText = `${m.cpu_percent}%`;
        }
        const gpuEl = document.getElementById('hudGpuVal');
        if (gpuEl) {
          const gpuPct = (m.gpu_percent !== null && m.gpu_percent !== undefined) ? m.gpu_percent : (data.progress && data.progress.gpu_busy);
          gpuEl.innerText = (gpuPct !== null && gpuPct !== undefined) ? `${gpuPct}%` : '0%';
        }
      }

      // 2. Queue HUD Chip Update
      if (data.queue) {
        currentQueueSnapshot = data.queue;
        const qEl = document.getElementById('hudQueueVal');
        if (qEl) {
          qEl.innerText = `${data.queue.total} pending`;
        }
      }

      // 3. Execution State Tracking
      if (data.status === 'running') {
        isRunning = true;
        badge.className = 'status-badge running';
        statusText.innerText = 'Running';
        const progBox = document.getElementById('progressBox');
        if (progBox) progBox.classList.add('active');

        // Keep button active to allow queueing the next pipeline task
        genBtn.disabled = false;
        genBtn.classList.add('enqueue-mode');
        genBtn.innerHTML = `<span>+ Enqueue Next Task (대기열 추가)</span>`;

        if (!clientTickerInterval && data.start_time) {
          startClientTicker(data.start_time);
        }

        const p = data.progress || {};
        document.getElementById('barFill').style.width = (p.percent || 0) + '%';
        document.getElementById('stepCounter').innerText = `${p.step || 0} / ${p.total_steps || 0} (${(p.percent || 0).toFixed(1)}%)`;
        document.getElementById('phaseText').innerText = p.message || p.phase || 'Executing...';

        // Live Node & Task Stage Tracker
        const nodeRow = document.getElementById('activeNodeRow');
        const nodeBadge = document.getElementById('activeNodeBadge');
        const nodeTitle = document.getElementById('activeNodeTitle');
        if (p.current_node) {
          nodeRow.style.display = 'flex';
          nodeBadge.innerText = `Node ${p.current_node.index || 1}/${p.current_node.total || 1}`;
          nodeTitle.innerText = `${p.current_node.title || p.current_node.class_type || 'Processing'}`;
        } else {
          nodeRow.style.display = 'none';
        }

        // Live DAG Node Pulse & Completion Highlighting
        highlightActiveNodesInDAG(p.current_node, p.executed_nodes || []);

        let speedStr = p.speed_s_per_it ? `${p.speed_s_per_it.toFixed(1)}s/it` : '';
        let etaStr = (p.eta_seconds !== null && p.eta_seconds !== undefined) ? `ETA: ${Math.round(p.eta_seconds)}s` : '';
        document.getElementById('speedEtaText').innerText = [speedStr, etaStr].filter(Boolean).join(' | ') || 'Executing pipeline...';
      } else if (data.status === 'complete') {
        isRunning = false;
        if (clientTickerInterval) { clearInterval(clientTickerInterval); clientTickerInterval = null; }
        badge.className = 'status-badge';
        statusText.innerText = 'Ready';
        genBtn.disabled = false;
        genBtn.classList.remove('enqueue-mode');
        switchTab(activeTab);

        clearNodeHighlights();
        document.getElementById('progressBox').classList.remove('active');

        if (data.result && data.result.url) {
          const prevBox = document.getElementById('previewBox');
          const img = document.getElementById('resultImg');
          img.src = data.result.url;
          document.getElementById('downloadLink').href = data.result.url;
          const elapsedSec = (data.result.elapsed_sec !== undefined) ? data.result.elapsed_sec : (data.elapsed || 0);
          const elapsedStr = `${Number(elapsedSec).toFixed(1)}s`;
          document.getElementById('elapsedText').innerText = `Elapsed: ${elapsedStr}`;
          const hudTime = document.getElementById('hudTimeVal');
          if (hudTime) hudTime.innerText = elapsedStr;
          prevBox.classList.add('active');
        }
      } else if (data.status === 'error') {
        isRunning = false;
        if (clientTickerInterval) { clearInterval(clientTickerInterval); clientTickerInterval = null; }
        badge.className = 'status-badge';
        statusText.innerText = 'Error';
        genBtn.disabled = false;
        genBtn.classList.remove('enqueue-mode');
        switchTab(activeTab);
        clearNodeHighlights();
        document.getElementById('progressBox').classList.remove('active');
        alert('Generation Failed: ' + (data.error || 'Unknown error'));
      } else if (data.status === 'idle') {
        isRunning = false;
        if (clientTickerInterval) { clearInterval(clientTickerInterval); clientTickerInterval = null; }
        badge.className = 'status-badge';
        statusText.innerText = 'Ready';
        genBtn.disabled = false;
        genBtn.classList.remove('enqueue-mode');
        switchTab(activeTab);
        clearNodeHighlights();

        // Restore last completed image and elapsed time on reload if available
        if (data.result && data.result.url) {
          const prevBox = document.getElementById('previewBox');
          const img = document.getElementById('resultImg');
          img.src = data.result.url;
          document.getElementById('downloadLink').href = data.result.url;
          const elapsedSec = (data.result.elapsed_sec !== undefined) ? data.result.elapsed_sec : (data.elapsed || 0);
          const elapsedStr = `${Number(elapsedSec).toFixed(1)}s`;
          document.getElementById('elapsedText').innerText = `Elapsed: ${elapsedStr}`;
          const hudTime = document.getElementById('hudTimeVal');
          if (hudTime) hudTime.innerText = elapsedStr;
          prevBox.classList.add('active');
        }
      }
    }

    function highlightActiveNodesInDAG(currentNode, executedNodes) {
      const activeId = currentNode ? String(currentNode.id) : null;
      const doneSet = new Set((executedNodes || []).map(x => String(x)));

      document.querySelectorAll('.flow-node-card').forEach(card => {
        const nid = String(card.dataset.nodeId);
        card.classList.toggle('flow-node-executing', nid === activeId);
        card.classList.toggle('flow-node-completed', doneSet.has(nid) && nid !== activeId);
      });
    }

    function clearNodeHighlights() {
      document.querySelectorAll('.flow-node-card').forEach(card => {
        card.classList.remove('flow-node-executing');
        card.classList.remove('flow-node-completed');
      });
    }

    // Pipeline Queue Modal Management
    function openQueueModal() {
      renderQueueList();
      document.getElementById('queueModal').style.display = 'flex';
    }

    function closeQueueModal() {
      document.getElementById('queueModal').style.display = 'none';
    }

    function renderQueueList() {
      const container = document.getElementById('queueListContainer');
      if (!container) return;
      if (!currentQueueSnapshot || !currentQueueSnapshot.items || currentQueueSnapshot.items.length === 0) {
        container.innerHTML = '<div style="font-size: 12px; color: var(--text-muted); text-align: center; padding: 14px;">현재 대기 중인 작업이 없습니다.</div>';
        return;
      }
      container.innerHTML = currentQueueSnapshot.items.map((job, idx) => `
        <div class="queue-item">
          <div style="min-width:0; flex:1;">
            <div style="font-size: 11px; font-weight: 700; color: #38bdf8;">#${idx + 1} [${job.type.toUpperCase()}]</div>
            <div style="font-size: 13px; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(job.name)}</div>
          </div>
          <button type="button" class="sheet-btn close" style="padding: 4px 8px; font-size: 11px;" onclick="cancelQueueItem('${job.id}')" title="대기열에서 제거">✕</button>
        </div>
      `).join('');
    }

    async function cancelQueueItem(jobId) {
      try {
        const resp = await fetch('/api/queue/cancel', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ job_id: jobId })
        });
        const res = await resp.json();
        if (res.success) {
          const qResp = await fetch('/api/queue');
          const qData = await qResp.json();
          currentQueueSnapshot = qData.queue;
          const qEl = document.getElementById('hudQueueVal');
          if (qEl) qEl.innerText = `${currentQueueSnapshot.total} pending`;
          renderQueueList();
        }
      } catch (err) {
        alert('Failed to cancel job: ' + err.message);
      }
    }

    async function clearQueue() {
      if (!confirm('대기열의 모든 작업을 취소하시겠습니까?')) return;
      try {
        const resp = await fetch('/api/queue/clear', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'}
        });
        const res = await resp.json();
        if (res.success) {
          currentQueueSnapshot = { total: 0, items: [] };
          const qEl = document.getElementById('hudQueueVal');
          if (qEl) qEl.innerText = '0 pending';
          renderQueueList();
        }
      } catch (err) {
        alert('Failed to clear queue: ' + err.message);
      }
    }

    // Idle Resource HUD Background Poller (every 2.5s)
    function startBackgroundHudPolling() {
      if (backgroundHudInterval) clearInterval(backgroundHudInterval);
      backgroundHudInterval = setInterval(async () => {
        if (isRunning) return;
        try {
          const resp = await fetch('/api/progress');
          const data = await resp.json();
          if (data.metrics) {
            const m = data.metrics;
            const ramEl = document.getElementById('hudRamVal');
            if (ramEl) ramEl.innerText = `${m.ram_used_gb}/${m.ram_total_gb} GB (${m.ram_percent}%)`;
            const cpuEl = document.getElementById('hudCpuVal');
            if (cpuEl) cpuEl.innerText = `${m.cpu_percent}%`;
            const gpuEl = document.getElementById('hudGpuVal');
            if (gpuEl) gpuEl.innerText = (m.gpu_percent !== null && m.gpu_percent !== undefined) ? `${m.gpu_percent}%` : '0%';
          }
          if (data.queue) {
            currentQueueSnapshot = data.queue;
            const qEl = document.getElementById('hudQueueVal');
            if (qEl) qEl.innerText = `${data.queue.total} pending`;
          }
          if (data.status === 'running') {
            updateUIWithProgress(data);
          }
        } catch (e) {}
      }, 2500);
    }

    async function probeInitialServerState() {
      try {
        const resp = await fetch('/api/progress');
        if (!resp.ok) return;
        const data = await resp.json();
        updateUIWithProgress(data);
        if (data.status === 'running') {
          startProgressPolling();
        }
      } catch (e) {
        console.warn('Initial server state probe warning:', e);
      }
    }

    function shareToGallery() {
      alert('Artifact synchronized to device storage: /sdcard/Pictures/TermuxDiffusion/');
    }

    // Auto-select standard workflow and initialize draggable sheet on load
    window.addEventListener('DOMContentLoaded', async () => {
      setupBottomSheetDrag();
      await loadDeviceModelCatalog();
      await loadPresetList();
      const savedWf = localStorage.getItem('amfy_selected_workflow') || localStorage.getItem('amvi_selected_workflow') || 'standard_photoreal';
      await selectPresetWorkflow(savedWf, null, false);
      startBackgroundHudPolling();
      await probeInitialServerState();
    });
  </script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


class DiffusionWebServer:
    """Production daemon wrapper for Termux Diffusion AmfyUI Mobile Server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 11553) -> None:
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        """Starts the HTTP server."""
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(line_buffering=True)
            except Exception:
                pass

        ThreadingHTTPServer.allow_reuse_address = True
        self.server = ThreadingHTTPServer((self.host, self.port), DiffusionWebHandler)
        actual_port = self.server.server_port
        logger.info("[DiffusionWebServer] AmfyUI Server live on http://%s:%d", self.host, actual_port)
        print(f"\n==================================================================")
        print(f"  AMEVA AmfyUI Mobile Studio Live (Galaxy S25/S21 Viewport)")
        print(f"  Local Device URL  : http://localhost:{actual_port}")
        print(f"  Network / Tailscale: http://{self.host}:{actual_port}")
        print(f"==================================================================\n")
        sys.stdout.flush()

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\n[DiffusionWebServer] Stopping server...")
                self.stop()
        else:
            self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stops the running HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("[DiffusionWebServer] Server terminated.")


def run_web_server(host: str = "0.0.0.0", port: int = 11553, blocking: bool = True) -> DiffusionWebServer:
    """Convenience entry point to launch the AmfyUI mobile web server."""
    server = DiffusionWebServer(host=host, port=port)
    server.start(blocking=blocking)
    return server
