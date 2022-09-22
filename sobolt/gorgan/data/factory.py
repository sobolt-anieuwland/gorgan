from typing import List, Dict, Any

from . import Dataset, ImageFolderDataset, RandomDataset, DatasetPair


def dataset_factory(
    dataset_config: Dict[str, Any], whole_config: Dict[str, Any], *args
) -> Dataset:
    if (
        isinstance(dataset_config, dict)
        and "originals" in dataset_config
        and "targets" in dataset_config
    ):
        normalization = whole_config.get("normalization", [0.0, 1.0])
        ds_originals = subdataset_factory(
            dataset_config["originals"], normalization, *args
        )
        ds_targets = subdataset_factory(dataset_config["targets"], normalization, *args)
        return DatasetPair(ds_originals, ds_targets)
    else:
        raise ValueError(
            "Malformed dataset definition: not a list or dictionary or missing originals / targets sections."
        )


def subdataset_factory(config: Dict[str, Any], normalization: List[float], *args):
    mapping = {"random-latent-vector": RandomDataset, "image-folder": ImageFolderDataset}
    dataset = mapping.get(config["type"])
    if dataset is None:
        raise ValueError("Failed finding dataset type")
    return dataset.from_config(config, normalization=normalization)  # type: ignore
