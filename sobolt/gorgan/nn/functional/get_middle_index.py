from typing import Dict, Tuple
import torch


def middle_index(dictionary: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    middle_lower_ix = (len(list(dictionary.keys())) // 2) - 1
    middle_upper_ix = (len(list(dictionary.keys())) // 2) + 1
    return middle_lower_ix, middle_upper_ix
