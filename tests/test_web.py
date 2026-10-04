import json
import urllib.request
import urllib.error
import pytest
from unittest.mock import patch, MagicMock

from termux_diffusion.web import DiffusionWebServer
from termux_diffusion.core import GenerationResult


@pytest.fixture
def test_server():
    # Use random free port
    server = DiffusionWebServer(host="127.0.0.1", port=0)
    server.start(blocking=False)
    actual_port = server.server.server_port
    base_url = f"http://127.0.0.1:{actual_port}"
    yield base_url
    server.stop()


def test_get_mobile_ui(test_server):
    req = urllib.request.Request(test_server + "/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "<meta name=\"viewport\"" in html
        assert "amviUI" in html
        assert "Quick Mode" in html
        assert "Generate" in html
        assert "FaceDetailer" in html


def test_api_models(test_server):
    req = urllib.request.Request(test_server + "/api/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "presets" in data
        assert "cached_models" in data
        assert "model_catalog" in data
        assert "hardware" in data


def test_api_status(test_server):
    req = urllib.request.Request(test_server + "/api/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "job" in data
        assert "system" in data


def test_api_workflow_inspect(test_server):
    wf_payload = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {}},
        "2": {"class_type": "SaveImage", "inputs": {"img": ["1", 0]}}
    }
    req = urllib.request.Request(
        test_server + "/api/workflow/inspect",
        data=json.dumps(wf_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["format"] == "API_PROMPT"
        assert data["total_nodes"] == 2
        assert len(data["execution_order"]) == 2
        assert data["execution_order"][0]["node_id"] == "1"
        assert data["execution_order"][1]["node_id"] == "2"


def test_api_workflows(test_server):
    req = urllib.request.Request(test_server + "/api/workflows")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "workflows" in data
        assert data["total"] >= 3
        ids = [w["id"] for w in data["workflows"]]
        assert "standard_photoreal" in ids
        assert "z_image_turbo" in ids
        assert "sdxs_lightning" in ids

    req_detail = urllib.request.Request(test_server + "/api/workflows/standard_photoreal")
    with urllib.request.urlopen(req_detail) as resp:
        assert resp.status == 200
        wf = json.loads(resp.read().decode("utf-8"))
        assert "nodes" in wf
        assert "links" in wf
        assert len(wf["nodes"]) == 8


def test_api_generate_dispatch(test_server):
    with patch("termux_diffusion.core.generate") as mock_gen:
        mock_gen.return_value = MagicMock(
            path=MagicMock(name="test.png"),
            elapsed_sec=1.2,
            device="cpu",
            steps=1,
            seed=42,
            width=512,
            height=512
        )
        gen_payload = {
            "prompt": "Test woman portrait",
            "model": "realistic",
            "steps": 1,
            "device": "cpu"
        }
        req = urllib.request.Request(
            test_server + "/api/generate",
            data=json.dumps(gen_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "running"


def test_server_default_port():
    server = DiffusionWebServer()
    assert server.port == 11553
    assert server.host == "0.0.0.0"


def test_cli_web_port_args():
    from termux_diffusion.cli import main

    # Test parser behavior with --port
    with patch("termux_diffusion.web.run_web_server") as mock_run:
        # Case 1: no --port specified
        main(["web"])
        mock_run.assert_called_with(host="0.0.0.0", port=11553, blocking=True)

        # Case 2: --port with no value
        main(["web", "--port"])
        mock_run.assert_called_with(host="0.0.0.0", port=11553, blocking=True)

        # Case 3: --port with explicit value
        main(["web", "--port", "19999"])
        mock_run.assert_called_with(host="0.0.0.0", port=19999, blocking=True)


def test_workflow_update_node_and_save_api(test_server, tmp_path, monkeypatch):
    import urllib.request
    import json
    from tests.test_workflow import SAMPLE_UI_WORKFLOW

    user_wf_dir = tmp_path / "user_workflows"
    monkeypatch.setattr("termux_diffusion.workflow.get_user_workflows_dir", lambda: user_wf_dir)

    # 1. Test update_node
    update_payload = {
        "workflow": SAMPLE_UI_WORKFLOW,
        "node_id": "3",
        "parameters": {"steps": 25, "cfg": 8.5}
    }
    req = urllib.request.Request(
        test_server + "/api/workflow/update_node",
        data=json.dumps(update_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["success"] is True
        assert res["inspection"]["nodes_detail"]["3"]["parameters"]["steps"] == 25
        assert res["inspection"]["nodes_detail"]["3"]["parameters"]["cfg"] == 8.5

    # 2. Test save workflow
    save_payload = {
        "workflow": res["workflow"],
        "id": "my_test_saved_flow",
        "name": "My Tuned Flow",
        "description": "Tuned workflow for mobile"
    }
    req = urllib.request.Request(
        test_server + "/api/workflow/save",
        data=json.dumps(save_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        save_res = json.loads(resp.read().decode("utf-8"))
        assert save_res["success"] is True
        assert save_res["saved"]["id"] == "my_test_saved_flow"

    # 3. Test export workflow
    export_url = test_server + "/api/workflows/my_test_saved_flow/export"
    with urllib.request.urlopen(export_url) as resp:
        assert resp.status == 200
        assert "attachment" in resp.headers.get("Content-Disposition", "")
        exported_data = json.loads(resp.read().decode("utf-8"))
        assert exported_data["name"] == "My Tuned Flow"


def test_api_queue_and_metrics(test_server):
    # Verify /api/queue
    req = urllib.request.Request(test_server + "/api/queue")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "queue" in data
        assert "total" in data["queue"]
        assert "items" in data["queue"]

    # Verify /api/progress returns metrics and queue
    req_prog = urllib.request.Request(test_server + "/api/progress")
    with urllib.request.urlopen(req_prog) as resp:
        assert resp.status == 200
        prog = json.loads(resp.read().decode("utf-8"))
        assert "metrics" in prog
        assert "ram_total_gb" in prog["metrics"]
        assert "cpu_percent" in prog["metrics"]
        assert "gpu_percent" in prog["metrics"]
        assert "queue" in prog


def test_queue_dispatch_cancel_and_clear(test_server, monkeypatch):
    from termux_diffusion import web

    # Simulate running job to test queue insertion
    with web._STATE_LOCK:
        web._CURRENT_JOB["status"] = "running"
        web._CURRENT_JOB["job_id"] = "running_test_job"

    try:
        # 1. Enqueue job 1
        payload1 = {"prompt": "Queue item 1", "model": "realistic"}
        req1 = urllib.request.Request(
            test_server + "/api/generate",
            data=json.dumps(payload1).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req1) as resp:
            assert resp.status == 202
            res1 = json.loads(resp.read().decode("utf-8"))
            assert res1["status"] == "queued"
            assert res1["queue_position"] == 1
            job1_id = res1["job_id"]

        # 2. Enqueue job 2
        payload2 = {"prompt": "Queue item 2", "model": "anime"}
        req2 = urllib.request.Request(
            test_server + "/api/generate",
            data=json.dumps(payload2).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req2) as resp:
            assert resp.status == 202
            res2 = json.loads(resp.read().decode("utf-8"))
            assert res2["status"] == "queued"
            assert res2["queue_position"] == 2
            job2_id = res2["job_id"]

        # 3. Check queue via API
        req_q = urllib.request.Request(test_server + "/api/queue")
        with urllib.request.urlopen(req_q) as resp:
            q_data = json.loads(resp.read().decode("utf-8"))
            assert q_data["queue"]["total"] == 2
            ids = [j["id"] for j in q_data["queue"]["items"]]
            assert job1_id in ids
            assert job2_id in ids

        # 4. Cancel job 1
        req_cancel = urllib.request.Request(
            test_server + "/api/queue/cancel",
            data=json.dumps({"job_id": job1_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_cancel) as resp:
            assert resp.status == 200
            cancel_res = json.loads(resp.read().decode("utf-8"))
            assert cancel_res["success"] is True

        # Check queue has 1 item
        with urllib.request.urlopen(req_q) as resp:
            q_data = json.loads(resp.read().decode("utf-8"))
            assert q_data["queue"]["total"] == 1
            assert q_data["queue"]["items"][0]["id"] == job2_id

        # 5. Clear queue
        req_clear = urllib.request.Request(
            test_server + "/api/queue/clear",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_clear) as resp:
            assert resp.status == 200
            clear_res = json.loads(resp.read().decode("utf-8"))
            assert clear_res["success"] is True

        # Check queue is empty
        with urllib.request.urlopen(req_q) as resp:
            q_data = json.loads(resp.read().decode("utf-8"))
            assert q_data["queue"]["total"] == 0
    finally:
        with web._STATE_LOCK:
            web._CURRENT_JOB["status"] = "idle"
            web._CURRENT_JOB["job_id"] = None
        web._clear_job_queue()



