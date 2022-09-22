from typing import Dict, Any, Iterator, List, Union

import torch.optim as optim
from torch.nn import Parameter


def optimizer_factory(
    parameters: Union[Iterator[Parameter], List[Parameter]],
    name: str,
    kwargs: Dict[str, Any],
):
    mappings = {
        "Adadelta": optim.Adadelta,
        "Adagrad": optim.Adagrad,
        "Adam": optim.Adam,
        "Adamax": optim.Adamax,
        "AdamW": optim.AdamW,
        "ASGD": optim.ASGD,
        "LBFGS": optim.LBFGS,
        "RMSprop": optim.RMSprop,
        "Rprop": optim.Rprop,
        "SGD": optim.SGD,
        "SparseAdam": optim.SparseAdam,
    }

    optimizer = mappings[name]
    return optimizer(parameters, **kwargs)
