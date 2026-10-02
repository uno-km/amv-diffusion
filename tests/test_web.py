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
        assert "AMEVA Termux Diffusion" in html
        assert "만들어! (Generate)" in html
        assert "FaceDetailer" in html


def test_api_models(test_server):
    req = urllib.request.Request(test_server + "/api/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "presets" in data
        assert "cached_models" in data
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
