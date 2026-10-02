"""Mobile-Optimized Web UI & Local Server for termux-diffusion.

Tailored specifically for smartphone viewports (e.g. Samsung Galaxy S25 / S21 / S20, 360px-420px width).
Zero third-party dependencies: Built strictly with Python standard library (http.server, threading).

Features:
1. Method 1 (Quick Generation): Single-touch "만들어! (Generate)" button with model presets, prompt chips, and sliders.
2. Method 2 (ComfyUI Workflow Runner): Upload ComfyUI UI Graph or API Prompt JSON, inspect DAG order, and execute.
3. Real-Time Telemetry: Live progress bar, step counter, speed, ETA, GPU load, and memory gauge.
4. One-touch Android Gallery sync and high-resolution image preview.
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
from .hub import list_presets, list_cached_models
from .hardware import detect_hardware_profile, get_kernel_physical_memory_gb
from .platform import is_android_termux
from .workflow import WorkflowExecutor, WorkflowParser, DAGEngine, WorkflowError

logger = logging.getLogger("termux_diffusion.web")

# Global generation state manager
_STATE_LOCK = threading.Lock()
_CURRENT_JOB: Dict[str, Any] = {
    "status": "idle",       # "idle", "running", "complete", "error"
    "progress": {
        "step": 0,
        "total_steps": 0,
        "percent": 0.0,
        "phase": "idle",
        "speed_s_per_it": None,
        "eta_seconds": None,
        "gpu_busy": None,
        "rss_mb": 0.0,
        "message": "Ready"
    },
    "result": None,
    "error": None,
    "start_time": 0.0,
    "elapsed": 0.0
}


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
        elif path == "/api/progress":
            with _STATE_LOCK:
                self._send_json(_CURRENT_JOB)
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
        elif path == "/api/workflow/run":
            self._handle_api_workflow_run(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "API route not found")

    def _serve_models_api(self) -> None:
        presets = list_presets()
        cached = list_cached_models()
        hw = detect_hardware_profile()
        ram_info = get_kernel_physical_memory_gb()
        ram_val = ram_info[0] if isinstance(ram_info, (tuple, list)) else ram_info

        self._send_json({
            "presets": presets,
            "cached_models": cached,
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

    def _serve_output_file(self, filename: str) -> None:
        safe_name = os.path.basename(filename)
        output_dirs = [
            Path.home() / ".cache" / "termux-diffusion" / "outputs",
            get_galaxy_gallery_dir(),
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
            if _CURRENT_JOB["status"] == "running":
                self._send_json({"error": "Generation already in progress", "status": "busy"}, 409)
                return
            _CURRENT_JOB["status"] = "running"
            _CURRENT_JOB["error"] = None
            _CURRENT_JOB["result"] = None
            _CURRENT_JOB["start_time"] = time.time()
            _CURRENT_JOB["elapsed"] = 0.0
            _CURRENT_JOB["progress"] = {
                "step": 0, "total_steps": 1, "percent": 0.0, "phase": "init",
                "speed_s_per_it": None, "eta_seconds": None, "gpu_busy": None, "rss_mb": 0.0,
                "message": "Initializing pipeline..."
            }

        def worker():
            try:
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
                    progress_callback=_update_progress,
                    progress=True
                )

                with _STATE_LOCK:
                    _CURRENT_JOB["status"] = "complete"
                    _CURRENT_JOB["elapsed"] = res.elapsed_sec
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

        t = threading.Thread(target=worker, daemon=True, name="DiffusionWorker")
        t.start()
        self._send_json({"message": "Generation dispatched", "status": "running"})

    def _handle_api_workflow_inspect(self, body: Dict[str, Any]) -> None:
        try:
            wf_data = body.get("workflow", body)
            nodes = WorkflowParser.parse(wf_data)
            order = DAGEngine.topological_sort(nodes)
            raw = WorkflowParser.load_json(wf_data) if not isinstance(wf_data, dict) else wf_data
            fmt = WorkflowParser.detect_format(raw)

            steps_info = []
            for idx, nid in enumerate(order, 1):
                n = nodes[nid]
                steps_info.append({
                    "step": idx,
                    "node_id": nid,
                    "class_type": n.class_type,
                    "title": n.title or n.class_type
                })

            self._send_json({
                "format": fmt,
                "total_nodes": len(nodes),
                "execution_order": steps_info
            })
        except WorkflowError as wfe:
            self._send_json({"error": str(wfe)}, 400)

    def _handle_api_workflow_run(self, body: Dict[str, Any]) -> None:
        with _STATE_LOCK:
            if _CURRENT_JOB["status"] == "running":
                self._send_json({"error": "A job is already running", "status": "busy"}, 409)
                return
            _CURRENT_JOB["status"] = "running"
            _CURRENT_JOB["error"] = None
            _CURRENT_JOB["result"] = None
            _CURRENT_JOB["start_time"] = time.time()
            _CURRENT_JOB["elapsed"] = 0.0

        def wf_worker():
            try:
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
                    progress=True
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

        t = threading.Thread(target=wf_worker, daemon=True, name="WorkflowWorker")
        t.start()
        self._send_json({"message": "Workflow dispatched", "status": "running"})

    def _serve_mobile_ui(self) -> None:
        """Renders the high-performance, responsive Galaxy S25 OLED dark mobile interface."""
        html = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>AMEVA Termux Diffusion</title>
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
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>AMEVA</span>
      <span class="brand-badge">Diffusion</span>
    </div>
    <div class="status-badge" id="statusBadge">
      <span class="status-dot"></span>
      <span id="statusText">대기 중</span>
    </div>
  </header>

  <main>
    <div class="tabs">
      <button class="tab-btn active" id="tabQuickBtn" onclick="switchTab('quick')">⚡ 직접 생성</button>
      <button class="tab-btn" id="tabWfBtn" onclick="switchTab('wf')">🔀 배전반 (ComfyUI)</button>
    </div>

    <!-- TAB 1: Quick Generate -->
    <div id="quickTab" style="display: flex; flex-direction: column; gap: 14px;">
      <div class="card">
        <div class="card-title">모델 선택</div>
        <div class="chip-scroll" id="modelChips">
          <div class="chip active" data-model="realistic" onclick="selectModel(this)">Realistic (실사)</div>
          <div class="chip" data-model="z-image-turbo" onclick="selectModel(this)">Z-Image Turbo (6B)</div>
          <div class="chip" data-model="sdxs" onclick="selectModel(this)">SDXS (초고속)</div>
          <div class="chip" data-model="anime" onclick="selectModel(this)">Anime (애니)</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">프롬프트</div>
        <textarea id="promptInput" rows="3" placeholder="생성하고 싶은 이미지 내용을 입력하세요...">A beautiful Korean woman in her 20s smiling warmly, radiant eyes, natural skin texture, 8k portrait photography</textarea>
        <div class="quick-tags">
          <span class="tag" onclick="addTag('Korean woman portrait')">+ 20대 한국 여성</span>
          <span class="tag" onclick="addTag('natural sunlight, soft daylight')">+ 자연광 채광</span>
          <span class="tag" onclick="addTag('8k masterpiece, raw photo')">+ 8K 실사화보</span>
          <span class="tag" onclick="addTag('street fashion in Seoul')">+ 서울 스트릿</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">세부 옵션</div>
        <div class="controls-grid">
          <div class="ctrl-group">
            <label>스텝 수: <span id="stepsVal">15</span></label>
            <input type="range" id="stepsInput" min="1" max="30" value="15" oninput="stepsVal.innerText=this.value">
          </div>
          <div class="ctrl-group">
            <label>CFG: <span id="cfgVal">7.0</span></label>
            <input type="range" id="cfgInput" min="1.0" max="15.0" step="0.5" value="7.0" oninput="cfgVal.innerText=this.value">
          </div>
        </div>

        <div class="switch-row" style="margin-top: 6px;">
          <span style="font-weight: 600; font-size: 14px;">✨ FaceDetailer (안면 보정)</span>
          <label class="switch">
            <input type="checkbox" id="detailerSwitch" checked>
            <span class="slider"></span>
          </label>
        </div>
      </div>
    </div>

    <!-- TAB 2: ComfyUI Workflow -->
    <div id="wfTab" style="display: none; flex-direction: column; gap: 14px;">
      <div class="card">
        <div class="card-title">ComfyUI 배전반 JSON 투입</div>
        <div class="drop-zone" onclick="document.getElementById('wfFileInput').click()">
          <span style="font-size: 28px;">📂</span>
          <span style="font-weight: 600;">ComfyUI .json 파일 선택</span>
          <span style="font-size: 12px; color: var(--text-muted);">UI 포맷 및 API 프롬프트 포맷 100% 자동 지원</span>
          <input type="file" id="wfFileInput" accept=".json" style="display: none;" onchange="handleWorkflowFile(this)">
        </div>
        <div id="wfInfo" style="display: none;">
          <div class="card-title" style="margin-top: 10px;">위상 정렬 순서 (<span id="wfStepCount">0</span>단계)</div>
          <div class="node-sequence" id="nodeSequence"></div>
        </div>
      </div>
    </div>

    <!-- Real-time Progress Display -->
    <div class="progress-box" id="progressBox">
      <div class="metrics-row">
        <span id="phaseText">디노이징 연산 중...</span>
        <span id="stepCounter">0 / 0</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" id="barFill"></div>
      </div>
      <div class="metrics-row">
        <span id="speedEtaText">연산 준비 중...</span>
        <span id="gpuText">GPU: --%</span>
      </div>
    </div>

    <!-- Image Preview Result -->
    <div class="preview-box" id="previewBox">
      <img id="resultImg" class="preview-img" alt="Generated Image">
      <div class="metrics-row" style="width: 100%;">
        <span id="elapsedText">소요 시간: --</span>
        <span style="color: var(--success); font-weight: 700;">완주 성공 (Exit 0)</span>
      </div>
      <div class="preview-actions">
        <a id="downloadLink" class="action-btn" download="ai_generated.png" target="_blank">📥 다운로드</a>
        <button class="action-btn" onclick="shareToGallery()">📸 갤러리 확인</button>
      </div>
    </div>
  </main>

  <div class="bottom-bar">
    <button class="generate-btn" id="generateBtn" onclick="onGenerateClick()">
      <span>🚀 만들어! (Generate)</span>
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
        genBtn.innerHTML = '<span>🚀 만들어! (Generate)</span>';
      } else {
        genBtn.innerHTML = '<span>🔀 배전반 실행! (Run Workflow)</span>';
      }
    }

    function selectModel(elem) {
      document.querySelectorAll('#modelChips .chip').forEach(c => c.classList.remove('active'));
      elem.classList.add('active');
      currentModel = elem.dataset.model;
      if (currentModel === 'z-image-turbo') {
        document.getElementById('stepsInput').value = 8;
        document.getElementById('stepsVal').innerText = 8;
      } else if (currentModel === 'sdxs') {
        document.getElementById('stepsInput').value = 2;
        document.getElementById('stepsVal').innerText = 2;
      } else {
        document.getElementById('stepsInput').value = 15;
        document.getElementById('stepsVal').innerText = 15;
      }
    }

    function addTag(text) {
      const p = document.getElementById('promptInput');
      p.value = p.value.trim() + ', ' + text;
    }

    function handleWorkflowFile(input) {
      const file = input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          loadedWorkflowData = JSON.parse(e.target.result);
          const resp = await fetch('/api/workflow/inspect', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(loadedWorkflowData)
          });
          const data = await resp.json();
          if (data.error) {
            alert('배전반 파싱 오류: ' + data.error);
            return;
          }
          document.getElementById('wfInfo').style.display = 'block';
          document.getElementById('wfStepCount').innerText = data.total_nodes;
          const seq = document.getElementById('nodeSequence');
          seq.innerHTML = '';
          data.execution_order.forEach(step => {
            const pill = document.createElement('div');
            pill.className = 'node-pill';
            pill.innerHTML = `<span class="node-num">${step.step}</span><span>[#${step.node_id}] ${step.title}</span>`;
            seq.appendChild(pill);
          });
        } catch (err) {
          alert('JSON 파일 해석 오류: ' + err.message);
        }
      };
      reader.readAsText(file);
    }

    async function onGenerateClick() {
      const genBtn = document.getElementById('generateBtn');
      genBtn.disabled = true;
      document.getElementById('previewBox').classList.remove('active');
      document.getElementById('progressBox').classList.add('active');

      if (activeTab === 'quick') {
        const payload = {
          prompt: document.getElementById('promptInput').value,
          model: currentModel,
          steps: parseInt(document.getElementById('stepsInput').value),
          cfg: parseFloat(document.getElementById('cfgInput').value),
          detail_faces: document.getElementById('detailerSwitch').checked,
          device: 'gpu'
        };
        await fetch('/api/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
      } else {
        if (!loadedWorkflowData) {
          alert('배전반 .json 파일을 먼저 선택해주세요.');
          genBtn.disabled = false;
          document.getElementById('progressBox').classList.remove('active');
          return;
        }
        await fetch('/api/workflow/run', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ workflow: loadedWorkflowData, device: 'gpu' })
        });
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
      }, 500);
    }

    function updateUIWithProgress(data) {
      const badge = document.getElementById('statusBadge');
      const statusText = document.getElementById('statusText');
      const genBtn = document.getElementById('generateBtn');

      if (data.status === 'running') {
        badge.className = 'status-badge running';
        statusText.innerText = '생성 중...';
        genBtn.disabled = true;
        genBtn.innerText = '🔄 연산 수행 중...';

        const p = data.progress;
        document.getElementById('barFill').style.width = p.percent + '%';
        document.getElementById('stepCounter').innerText = `${p.step} / ${p.total_steps} (${p.percent.toFixed(1)}%)`;
        document.getElementById('phaseText').innerText = p.message || p.phase;
        
        let speedStr = p.speed_s_per_it ? `${p.speed_s_per_it.toFixed(1)}s/it` : '';
        let etaStr = p.eta_seconds !== null ? `ETA: ${Math.round(p.eta_seconds)}s` : '';
        document.getElementById('speedEtaText').innerText = [speedStr, etaStr].filter(Boolean).join(' | ') || '연산 진행 중...';
        document.getElementById('gpuText').innerText = p.gpu_busy !== null ? `GPU: ${p.gpu_busy}%` : '';
      } else if (data.status === 'complete') {
        clearInterval(pollInterval);
        badge.className = 'status-badge';
        statusText.innerText = '대기 중';
        genBtn.disabled = false;
        switchTab(activeTab);

        document.getElementById('progressBox').classList.remove('active');
        if (data.result && data.result.url) {
          const prevBox = document.getElementById('previewBox');
          const img = document.getElementById('resultImg');
          img.src = data.result.url + '?t=' + Date.now();
          document.getElementById('downloadLink').href = data.result.url;
          document.getElementById('elapsedText').innerText = `소요 시간: ${data.result.elapsed_sec.toFixed(1)}초`;
          prevBox.classList.add('active');
        }
      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        badge.className = 'status-badge';
        statusText.innerText = '오류 발생';
        genBtn.disabled = false;
        switchTab(activeTab);
        document.getElementById('progressBox').classList.remove('active');
        alert('생성 실패: ' + (data.error || '알 수 없는 오류'));
      }
    }

    function shareToGallery() {
      alert('산출물이 단말 갤러리(/sdcard/Pictures/TermuxDiffusion/)에 자동 동기화되었습니다.');
    }
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
    """Production daemon wrapper for Termux Diffusion Mobile Web UI."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = True) -> None:
        """Starts the HTTP server."""
        self.server = ThreadingHTTPServer((self.host, self.port), DiffusionWebHandler)
        actual_port = self.server.server_port
        logger.info("[DiffusionWebServer] Mobile Web UI live on http://%s:%d", self.host, actual_port)
        print(f"\n==================================================================")
        print(f"  AMEVA Termux Diffusion Mobile Web UI Live (Galaxy S25 Viewport)")
        print(f"  Local Device URL  : http://localhost:{actual_port}")
        print(f"  Network / Tailscale: http://{self.host}:{actual_port}")
        print(f"==================================================================\n")

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


def run_web_server(host: str = "0.0.0.0", port: int = 8080, blocking: bool = True) -> DiffusionWebServer:
    """Convenience entry point to launch the mobile web server."""
    server = DiffusionWebServer(host=host, port=port)
    server.start(blocking=blocking)
    return server
