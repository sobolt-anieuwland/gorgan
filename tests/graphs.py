import pytest

import torch
import torch.nn as nn

from sobolt.gorgan.graphs import graph_factory


@pytest.mark.parametrize(
    "graph_type",
    [
        "SpadeSisrGenerator", "ProgressiveSpadeGenerator", "EsrganProgressiveGenerator",
        "ResidualChannelAttentionGenerator", "LinearGenerator", "AlternativeEsrganGenerator",
    ],
)
@pytest.mark.parametrize(
    "shape_originals_gen", [[3, 16, 16], [4, 16, 16]],
)  # Small shape for testing in RAM-limited envs
def test_generator_graph_creation(graph_type: str, shape_originals_gen):
    graph_args = {
        "shape_originals": shape_originals_gen,
        "upsample_factor": 4,
        "use_progressive": True,
        "use_attention": True,
        "use_condition": False,
        "conditional_mask_indices": [],
        "init_prog_step": 2,
        "n_rrdb": 12,
    }

    graph_def = {"type": graph_type, "arguments": graph_args}

    # Checks graph type is nn.Module
    graph = graph_factory(graph_def)
    assert isinstance(graph, nn.Module)

    # Checks if graph return is a dict with torch.Tensor values
    tensor = torch.rand([1] + shape_originals_gen, dtype=torch.float32)
    result = graph(tensor)
    assert isinstance(result, dict) and isinstance(result["generated"], torch.Tensor)


@pytest.mark.parametrize(
    "graph_type",
    [
        "DcganDiscriminator", "EsrganDiscriminator", "LinearDiscriminator",
    ],
)
@pytest.mark.parametrize(
    "shape_originals", [[3, 128, 128], [4, 128, 128]],
)
@pytest.mark.parametrize(
    "shape_targets", [[3, 512, 512], [4, 512, 512]],
)
def test_discriminator_graph_creation(graph_type: str, shape_originals, shape_targets):
    graph_args = {"shape_originals": shape_originals, "shape_targets": shape_targets}

    # Checks graph type is nn.Module
    graph_def = {"type": graph_type, "arguments": graph_args}
    graph = graph_factory(graph_def)
    assert isinstance(graph, nn.Module)

    # Checks if graph return is a dict with torch.Tensor values
    tensor = torch.rand([1] + shape_targets, dtype=torch.float32)
    result = graph(tensor)
    assert isinstance(result, dict) and isinstance(result["discriminated"], torch.Tensor)


@pytest.mark.parametrize(
    "graph_type", ["UNet", "AlternativeEsrganGenerator", "EsrganProgressiveGenerator"],
)
def test_gradients(graph_type):
    # This test randomly fails sometimes. Nonetheless, let's keep it in to investigate
    # why. For example, if it turns out it is always a specific generator that fails, that
    # is important to know. If it is arbitrary which one, the test is probably just bad,
    # however.

    graph_args = {}
    if graph_type in ["AlternativeEsrganGenerator", "EsrganProgressiveGenerator"]:
        graph_args = {
            "shape_originals": [3, 16, 16],
            "upsample_factor": 4,
            "use_progressive": True,
            "use_attention": True,
            "use_condition": False,
            "conditional_mask_indices": [],
            "n_rrdb": 12,
        }
    elif graph_type == "UNet":
        graph_args = {
            "shape_originals": [3, 64, 64],
            "in_channels": 3,
            "out_channels": 3,
        }

    graph_def = {"type": graph_type, "arguments": graph_args}

    original_shape = graph_args["shape_originals"]
    graph = graph_factory(graph_def)
    optim = torch.optim.Adam(graph.parameters())
    tensor = torch.rand([1] + original_shape, dtype=torch.float32)

    result = graph(tensor)
    loss = result["generated"].mean()
    loss.backward()
    optim.step()

    # Checks whether a gradients are updated after an optimization step
    for param_name, param in graph.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, print("failed for: {}, layer param is none ".format(param_name))
            assert torch.sum(param.grad ** 2) != 0.0, print("failed for: {}, layer param is 0 ".format(param_name))
