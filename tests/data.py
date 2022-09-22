from pathlib import Path
import pytest

import torch

from sobolt.gorgan.data import RandomDataset, ImageFolderDataset, dataset_factory, DatasetPair


__random_dict = dict(
    type="random-latent-vector",
    args=dict(latent_length=100, render_dims=[64, 64]),
)


@pytest.mark.parametrize("latent_shape", [100, 103, 200])
def test_init_random_direct(latent_shape):
    render_shape = [128, 128]
    ds = RandomDataset(latent_shape, render_shape)
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert len(item.shape) == 3
    assert item.shape[0] == latent_shape


def test_init_random_from_dict():
    ds = RandomDataset.from_config(__random_dict)
    shape = __random_dict["args"]["latent_length"]
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert len(item.shape) == 3
    assert item.shape[0] == shape


@pytest.mark.parametrize("latent_shape", [100, 103, 200])
def test_render_random(latent_shape):
    render_shape = [128, 128]
    ds = RandomDataset(latent_shape, render_shape)
    item = ds[0]
    render = ds.render(item)
    dims = [3]
    dims.extend(render_shape)
    assert isinstance(render, torch.Tensor)
    assert len(render.shape) == 3
    assert render.shape[0] == 3
    for a, b in zip(render.shape, dims):
        assert a == b

def test_random_too_large_fails():
    with pytest.raises(AssertionError) as e:
        ds = RandomDataset(10 + 128 * 128, [128, 128])


__image_folder_path = Path(__file__).parent / "data" / "data-image-folder"
__image_folder_dict = dict(
    type="image-folder",
    args=dict(root=__image_folder_path),
)

def test_init_image_folder_direct():
    ds = ImageFolderDataset(__image_folder_path, [])
    assert len(ds) == 5
    assert isinstance(ds[0], torch.Tensor)
    assert ds[0].shape[0] == 3


def test_init_image_folder_config():
    ds = ImageFolderDataset.from_config(__image_folder_dict)
    assert len(ds) == 5
    assert isinstance(ds[0], torch.Tensor)
    assert ds[0].shape[0] == 3


__config = dict(
    train=dict(originals=__random_dict, targets=__image_folder_dict),
    validation=dict(originals=__random_dict, targets=__image_folder_dict)
)

def test_init_img_generation_dataset_pair():
    ds = dataset_factory(__config["train"])
    assert isinstance(ds, DatasetPair)
    ds = dataset_factory(__config["validation"])
    assert isinstance(ds, DatasetPair)
