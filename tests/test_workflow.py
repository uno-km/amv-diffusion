import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from termux_diffusion.workflow import (
    WorkflowParser,
    DAGEngine,
    WorkflowExecutor,
    WorkflowError,
    WorkflowFormatError,
    WorkflowCycleError,
    UnsupportedNodeError,
    execute_workflow,
    NODE_REGISTRY,
    NodeConnection,
)
from termux_diffusion.core import GenerationResult


SAMPLE_API_WORKFLOW = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "realistic.gguf"}
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": "A beautiful Korean woman portrait"}
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": "blurry, low quality"}
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"batch_size": 1, "height": 512, "width": 512}
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7.0,
            "denoise": 1.0,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler_a",
            "scheduler": "karras",
            "seed": 42,
            "steps": 15
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}
    }
}


SAMPLE_UI_WORKFLOW = {
    "last_node_id": 9,
    "last_link_id": 10,
    "nodes": [
        {
            "id": 4,
            "type": "CheckpointLoaderSimple",
            "inputs": [],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0},
                {"name": "CLIP", "type": "CLIP", "links": [3, 4], "slot_index": 1},
                {"name": "VAE", "type": "VAE", "links": [8], "slot_index": 2}
            ],
            "widgets_values": ["realistic.gguf"]
        },
        {
            "id": 6,
            "type": "CLIPTextEncode",
            "inputs": [{"name": "clip", "type": "CLIP", "link": 3}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5], "slot_index": 0}],
            "widgets_values": ["A stunning Korean portrait, 8k"]
        },
        {
            "id": 7,
            "type": "CLIPTextEncode",
            "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
            "widgets_values": ["blurry, ugly"]
        },
        {
            "id": 5,
            "type": "EmptyLatentImage",
            "inputs": [],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [2], "slot_index": 0}],
            "widgets_values": [512, 512, 1]
        },
        {
            "id": 3,
            "type": "KSampler",
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "positive", "type": "CONDITIONING", "link": 5},
                {"name": "negative", "type": "CONDITIONING", "link": 6},
                {"name": "latent_image", "type": "LATENT", "link": 2}
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7], "slot_index": 0}],
            "widgets_values": [1234567, "randomize", 20, 8.0, "euler_a", "karras", 1.0]
        },
        {
            "id": 8,
            "type": "VAEDecode",
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 7},
                {"name": "vae", "type": "VAE", "link": 8}
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}],
            "widgets_values": []
        },
        {
            "id": 9,
            "type": "SaveImage",
            "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
            "outputs": [],
            "widgets_values": ["ComfyOutput"]
        }
    ],
    "links": [
        [1, 4, 0, 3, 0, "MODEL"],
        [2, 5, 0, 3, 3, "LATENT"],
        [3, 4, 1, 6, 0, "CLIP"],
        [4, 4, 1, 7, 0, "CLIP"],
        [5, 6, 0, 3, 1, "CONDITIONING"],
        [6, 7, 0, 3, 2, "CONDITIONING"],
        [7, 3, 0, 8, 0, "LATENT"],
        [8, 4, 2, 8, 1, "VAE"],
        [9, 8, 0, 9, 0, "IMAGE"]
    ]
}


def test_detect_format():
    assert WorkflowParser.detect_format(SAMPLE_API_WORKFLOW) == "API_PROMPT"
    assert WorkflowParser.detect_format(SAMPLE_UI_WORKFLOW) == "UI_GRAPH"
    with pytest.raises(WorkflowFormatError):
        WorkflowParser.detect_format({"random_key": "invalid_value"})


def test_parse_api_prompt():
    nodes = WorkflowParser.parse(SAMPLE_API_WORKFLOW)
    assert len(nodes) == 7
    assert "3" in nodes
    assert nodes["3"].class_type == "KSampler"
    assert isinstance(nodes["3"].inputs["model"], NodeConnection)
    assert nodes["3"].inputs["model"].source_node_id == "4"


def test_parse_ui_graph():
    nodes = WorkflowParser.parse(SAMPLE_UI_WORKFLOW)
    assert len(nodes) == 7
    assert "4" in nodes
    assert nodes["4"].class_type == "CheckpointLoaderSimple"
    assert nodes["6"].widgets_values == ["A stunning Korean portrait, 8k"]
    # Check link connection on KSampler (id=3)
    ksampler = nodes["3"]
    assert "model" in ksampler.inputs
    assert ksampler.inputs["model"].source_node_id == "4"
    assert ksampler.inputs["positive"].source_node_id == "6"


def test_topological_sort_order():
    nodes = WorkflowParser.parse(SAMPLE_API_WORKFLOW)
    order = DAGEngine.topological_sort(nodes)
    assert len(order) == 7
    # Node 4 (loader) and 5 (latent) must precede Node 3 (sampler)
    assert order.index("4") < order.index("3")
    assert order.index("5") < order.index("3")
    # Node 6 and 7 (prompt) must precede Node 3
    assert order.index("6") < order.index("3")
    assert order.index("7") < order.index("3")
    # Node 3 must precede Node 8 (decode)
    assert order.index("3") < order.index("8")
    # Node 8 must precede Node 9 (save)
    assert order.index("8") < order.index("9")


def test_cycle_detection():
    cyclic_nodes = {
        "1": {"class_type": "A", "inputs": {"dep": ["2", 0]}},
        "2": {"class_type": "B", "inputs": {"dep": ["1", 0]}},
    }
    nodes = WorkflowParser.parse(cyclic_nodes)
    with pytest.raises(WorkflowCycleError):
        DAGEngine.topological_sort(nodes)


def test_unsupported_node_handling(tmp_path):
    unsupported_wf = {
        "1": {"class_type": "ExoticCustomNodeNotInRegistry", "inputs": {}}
    }
    executor = WorkflowExecutor()
    with pytest.raises(UnsupportedNodeError):
        executor.run(unsupported_wf, output_dir=tmp_path, force_run=False)

    # With force_run=True, should skip unregistered node without fatal crash
    res = executor.run(unsupported_wf, output_dir=tmp_path, force_run=True)
    assert res.success is True


def test_end_to_end_mock_execution(tmp_path):
    mock_img = tmp_path / "mock_output.png"
    mock_img.write_bytes(b"PNG_DATA")

    mock_result = GenerationResult(
        path=mock_img,
        gallery_path=None,
        prompt="A beautiful Korean woman portrait",
        negative_prompt=None,
        model="realistic",
        device="cpu",
        steps=15,
        cfg_scale=7.0,
        width=512,
        height=512,
        seed=42,
        elapsed_sec=1.5
    )

    with patch("termux_diffusion.core.generate", return_value=mock_result) as mock_gen:
        res = execute_workflow(SAMPLE_API_WORKFLOW, output_dir=tmp_path, device="cpu")
        assert res.success is True
        assert len(res.output_images) == 1
        assert res.output_images[0] == mock_img
        mock_gen.assert_called_once()
        assert mock_gen.call_args[1]["steps"] == 15
        assert mock_gen.call_args[1]["cfg_scale"] == 7.0
        assert mock_gen.call_args[1]["sampling_method"] == "euler_a"
        assert mock_gen.call_args[1]["schedule"] == "karras"
