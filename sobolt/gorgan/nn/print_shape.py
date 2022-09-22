import torch.nn as nn


class PrintShape(nn.Module):
    def __init__(self, msg: str = ""):
        super().__init__()
        self.__msg = " - {msg}" if msg != "" else ""

    def forward(self, inputs):
        print(f"Shape{self.__msg}:", inputs.shape)
        return inputs
