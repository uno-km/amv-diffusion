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


def test_inspect_workflow_details():
    from termux_diffusion.workflow import inspect_workflow_details

    res = inspect_workflow_details(SAMPLE_UI_WORKFLOW)
    assert res["format"] == "UI_GRAPH"
    assert res["total_nodes"] == 7
    assert "nodes_detail" in res
    assert "edges" in res
    assert len(res["edges"]) >= 6

    # Verify KSampler detail
    ksampler_detail = res["nodes_detail"]["3"]
    assert ksampler_detail["class_type"] == "KSampler"
    assert "denoises" in ksampler_detail["role"].lower()
    assert ksampler_detail["parameters"]["steps"] == 20
    assert ksampler_detail["parameters"]["cfg"] == 8.0
    assert len(ksampler_detail["connections_in"]) == 4

    # Verify CheckpointLoader detail
    ckpt_detail = res["nodes_detail"]["4"]
    assert ckpt_detail["parameters"]["ckpt_name"] == "realistic.gguf"
    assert len(ckpt_detail["connections_out"]) >= 2


def test_update_workflow_node_parameter_ui():
    import copy
    from termux_diffusion.workflow import update_workflow_node_parameter, inspect_workflow_details

    wf = copy.deepcopy(SAMPLE_UI_WORKFLOW)
    # Update KSampler (node 3)
    update_workflow_node_parameter(wf, "3", {
        "steps": 28,
        "cfg": 9.5,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 0.8
    })

    # Update CLIPTextEncode (node 6)
    update_workflow_node_parameter(wf, "6", {
        "text": "Modified positive prompt for photorealism"
    })

    inspected = inspect_workflow_details(wf)
    k_params = inspected["nodes_detail"]["3"]["parameters"]
    assert k_params["steps"] == 28
    assert k_params["cfg"] == 9.5
    assert k_params["sampler_name"] == "dpmpp_2m"
    assert k_params["denoise"] == 0.8

    clip_params = inspected["nodes_detail"]["6"]["parameters"]
    assert clip_params["text"] == "Modified positive prompt for photorealism"


def test_update_workflow_node_parameter_api():
    import copy
    from termux_diffusion.workflow import update_workflow_node_parameter, inspect_workflow_details

    wf = copy.deepcopy(SAMPLE_API_WORKFLOW)
    update_workflow_node_parameter(wf, "3", {
        "steps": 30,
        "cfg": 6.5
    })
    update_workflow_node_parameter(wf, "6", {
        "text": "API prompt updated"
    })

    inspected = inspect_workflow_details(wf)
    assert inspected["nodes_detail"]["3"]["parameters"]["steps"] == 30
    assert inspected["nodes_detail"]["3"]["parameters"]["cfg"] == 6.5
    assert inspected["nodes_detail"]["6"]["parameters"]["text"] == "API prompt updated"


def test_save_and_list_user_workflow(tmp_path, monkeypatch):
    import copy
    from termux_diffusion.workflow import (
        save_user_workflow,
        list_builtin_workflows,
        get_builtin_workflow
    )

    user_wf_dir = tmp_path / "user_workflows"
    monkeypatch.setattr("termux_diffusion.workflow.get_user_workflows_dir", lambda: user_wf_dir)

    wf = copy.deepcopy(SAMPLE_UI_WORKFLOW)
    saved = save_user_workflow(
        wf,
        workflow_id="test_custom_preset",
        name="Custom Preset Alpha",
        description="User custom tuned flow"
    )

    assert saved["id"] == "test_custom_preset"
    assert Path(saved["path"]).is_file()

    # Verify listing includes it
    all_wfs = list_builtin_workflows()
    found = next((w for w in all_wfs if w["id"] == "test_custom_preset"), None)
    assert found is not None
    assert found["name"] == "Custom Preset Alpha"
    assert found["custom"] is True

    # Verify get_builtin_workflow retrieves it
    retrieved = get_builtin_workflow("test_custom_preset")
    assert retrieved is not None
    assert retrieved["name"] == "Custom Preset Alpha"


def test_subgraph_inlining_and_topological_sort():
    """Validates ComfyUI v0.4 definitions.subgraphs inlining and Kahn's topological sort."""
    subgraph_wf = {
        "nodes": [
            {
                "id": 10,
                "type": "MarkdownNote",
                "inputs": [],
                "widgets_values": ["Documentation note"]
            },
            {
                "id": 20,
                "type": "subgraph-uuid-1234",
                "inputs": [
                    {"name": "prompt_text", "link": None},
                    {"name": "width", "link": 1}
                ],
                "widgets_values_named": {
                    "prompt_text": "Surreal fashion photography",
                    "unet_name": "z_image_bf16.safetensors"
                },
                "widgets_values": ["Surreal fashion photography", "z_image_bf16.safetensors"]
            },
            {
                "id": 30,
                "type": "ResolutionSelector",
                "inputs": [],
                "widgets_values": ["1:1 (Square)"]
            },
            {
                "id": 40,
                "type": "SaveImageAdvanced",
                "inputs": [{"name": "images", "link": 2}],
                "widgets_values": ["output_prefix", "png"]
            }
        ],
        "links": [
            [1, 30, 0, 20, 1, "INT"],
            [2, 20, 0, 40, 0, "IMAGE"]
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "subgraph-uuid-1234",
                    "inputs": [
                        {"name": "prompt_text", "linkIds": [101]},
                        {"name": "width", "linkIds": [102]},
                        {"name": "unet_name", "linkIds": [103]}
                    ],
                    "outputs": [
                        {"name": "IMAGE", "linkIds": [104]}
                    ],
                    "nodes": [
                        {
                            "id": 21,
                            "type": "UNETLoader",
                            "inputs": [{"name": "unet_name", "link": 103}],
                            "widgets_values": []
                        },
                        {
                            "id": 22,
                            "type": "CLIPTextEncode",
                            "inputs": [{"name": "text", "link": 101}],
                            "widgets_values": []
                        },
                        {
                            "id": 23,
                            "type": "EmptySD3LatentImage",
                            "inputs": [{"name": "width", "link": 102}],
                            "widgets_values": [512, 512]
                        },
                        {
                            "id": 24,
                            "type": "KSampler",
                            "inputs": [
                                {"name": "model", "link": 201},
                                {"name": "positive", "link": 202},
                                {"name": "latent_image", "link": 203}
                            ],
                            "widgets_values": [42, "randomize", 20, 4.0, "euler", "simple", 1.0]
                        },
                        {
                            "id": 25,
                            "type": "VAEDecode",
                            "inputs": [{"name": "samples", "link": 204}],
                            "widgets_values": []
                        }
                    ],
                    "links": [
                        {"id": 101, "origin_id": -10, "origin_slot": 0, "target_id": 22, "target_slot": 0, "type": "STRING"},
                        {"id": 102, "origin_id": -10, "origin_slot": 1, "target_id": 23, "target_slot": 0, "type": "INT"},
                        {"id": 103, "origin_id": -10, "origin_slot": 2, "target_id": 21, "target_slot": 0, "type": "COMBO"},
                        {"id": 201, "origin_id": 21, "origin_slot": 0, "target_id": 24, "target_slot": 0, "type": "MODEL"},
                        {"id": 202, "origin_id": 22, "origin_slot": 0, "target_id": 24, "target_slot": 1, "type": "CONDITIONING"},
                        {"id": 203, "origin_id": 23, "origin_slot": 0, "target_id": 24, "target_slot": 2, "type": "LATENT"},
                        {"id": 204, "origin_id": 24, "origin_slot": 0, "target_id": 25, "target_slot": 0, "type": "LATENT"},
                        {"id": 104, "origin_id": 25, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"}
                    ]
                }
            ]
        }
    }

    nodes = WorkflowParser.parse(subgraph_wf)
    # The wrapper node (20) was replaced by 5 subgraph nodes (21, 22, 23, 24, 25)
    # Root had: 10, 30, 40 -> Total: 8 nodes
    assert len(nodes) == 8
    assert "20" not in nodes
    assert "21" in nodes
    assert "22" in nodes
    assert "23" in nodes
    assert "24" in nodes
    assert "25" in nodes

    # Verify link rerouting: ResolutionSelector (30) slot 0 -> EmptySD3LatentImage (23) slot 0 (width)
    assert "width" in nodes["23"].inputs
    assert isinstance(nodes["23"].inputs["width"], NodeConnection)
    assert nodes["23"].inputs["width"].source_node_id == "30"

    # Verify link rerouting: VAEDecode (25) -> SaveImageAdvanced (40)
    assert "images" in nodes["40"].inputs
    assert isinstance(nodes["40"].inputs["images"], NodeConnection)
    assert nodes["40"].inputs["images"].source_node_id == "25"

    # Verify topological order
    order = DAGEngine.topological_sort(nodes)
    assert len(order) == 8
    assert order.index("30") < order.index("23")
    assert order.index("21") < order.index("24")
    assert order.index("22") < order.index("24")
    assert order.index("23") < order.index("24")
    assert order.index("24") < order.index("25")
    assert order.index("25") < order.index("40")


def test_zero_silent_fallback_missing_weights(tmp_path):
    """Zero-Silent-Fallback Protocol: Missing weights must immediately raise FileNotFoundError."""
    missing_wf = {
        "nodes": [
            {
                "id": 1,
                "type": "UNETLoader",
                "inputs": {},
                "widgets_values": ["non_existent_dit_model.safetensors"]
            }
        ],
        "links": []
    }

    executor = WorkflowExecutor()
    with pytest.raises(FileNotFoundError) as exc_info:
        executor.run(missing_wf, output_dir=tmp_path, force_run=False)

    assert "non_existent_dit_model.safetensors" in str(exc_info.value)
    assert "Zero-Silent-Fallback Protocol" in str(exc_info.value)


def test_z_image_turbo_native_flat_workflow(tmp_path):
    """Verify Comfy-Org official native flat Z-Image-Turbo workflow with ConditioningZeroOut."""
    wf_data = {
        "id": "9ae6082b-c7f4-433c-9971-7a8f65a3ea65",
        "nodes": [
            {"id": 62, "type": "CLIPLoader", "inputs": [], "widgets_values": ["qwen_3_4b_fp8_mixed.safetensors", "lumina2", "default"]},
            {"id": 63, "type": "VAELoader", "inputs": [], "widgets_values": ["ae.safetensors"]},
            {"id": 64, "type": "ConditioningZeroOut", "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 76}]},
            {"id": 65, "type": "VAEDecode", "inputs": [{"name": "samples", "type": "LATENT", "link": 78}, {"name": "vae", "type": "VAE", "link": 79}]},
            {"id": 66, "type": "UNETLoader", "inputs": [], "widgets_values": ["z_image_turbo_int8_convrot.safetensors", "default"]},
            {"id": 69, "type": "ModelSamplingAuraFlow", "inputs": [{"name": "model", "type": "MODEL", "link": 77}], "widgets_values": [3]},
            {"id": 70, "type": "KSampler", "inputs": [
                {"name": "model", "type": "MODEL", "link": 80},
                {"name": "positive", "type": "CONDITIONING", "link": 81},
                {"name": "negative", "type": "CONDITIONING", "link": 82},
                {"name": "latent_image", "type": "LATENT", "link": 83}
            ], "widgets_values": [121725701057393, "randomize", 8, 1, "res_multistep", "simple", 1]},
            {"id": 67, "type": "CLIPTextEncode", "inputs": [{"name": "clip", "type": "CLIP", "link": 84}], "widgets_values": ["Dramatic black and white high fashion studio portrait"]},
            {"id": 9, "type": "SaveImage", "inputs": [{"name": "images", "type": "IMAGE", "link": 85}], "widgets_values": ["z-image-turbo"]},
            {"id": 68, "type": "EmptySD3LatentImage", "inputs": [], "widgets_values": [1024, 1024, 1]}
        ],
        "links": [
            [76, 67, 0, 64, 0, "CONDITIONING"],
            [77, 66, 0, 69, 0, "MODEL"],
            [78, 70, 0, 65, 0, "LATENT"],
            [79, 63, 0, 65, 1, "VAE"],
            [80, 69, 0, 70, 0, "MODEL"],
            [81, 67, 0, 70, 1, "CONDITIONING"],
            [82, 64, 0, 70, 2, "CONDITIONING"],
            [83, 68, 0, 70, 3, "LATENT"],
            [84, 62, 0, 67, 0, "CLIP"],
            [85, 65, 0, 9, 0, "IMAGE"]
        ]
    }

    nodes = WorkflowParser.parse(wf_data)
    assert len(nodes) == 10
    order = DAGEngine.topological_sort(nodes)
    assert len(order) == 10
    assert order.index("67") < order.index("64")
    assert order.index("66") < order.index("69")
    assert order.index("64") < order.index("70")
    assert order.index("69") < order.index("70")
    assert order.index("70") < order.index("65")
    assert order.index("65") < order.index("9")

    mock_img = tmp_path / "test_turbo.png"
    mock_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch("termux_diffusion.workflow.handle_checkpoint_loader") as mock_ckpt, \
         patch("termux_diffusion.core.generate") as mock_gen:
        mock_ckpt.return_value = {"model": "z_image_turbo", "clip": "qwen_3_4b", "vae": "ae"}
        mock_gen.return_value = GenerationResult(
            path=mock_img,
            gallery_path=None,
            prompt="Dramatic black and white high fashion studio portrait",
            negative_prompt=None,
            model="z_image_turbo",
            device="cpu",
            steps=8,
            cfg_scale=1.0,
            width=1024,
            height=1024,
            seed=121725701057393,
            elapsed_sec=2.5
        )

        executor = WorkflowExecutor()
        res = executor.run(wf_data, output_dir=tmp_path, force_run=True)

        assert res.success is True
        assert len(res.output_images) == 1
        assert "70" in res.executed_nodes
        assert "64" in res.executed_nodes
        # Verify KSampler received CFG=1 and steps=8
        mock_gen.assert_called_once()
        _, kwargs = mock_gen.call_args
        assert kwargs["steps"] == 8
        assert kwargs["cfg_scale"] == 1.0
        assert kwargs["width"] == 1024
        assert kwargs["height"] == 1024




