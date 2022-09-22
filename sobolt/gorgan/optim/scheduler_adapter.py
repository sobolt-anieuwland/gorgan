from .lr_scheduler_threshold import LrThresholdScheduler
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
)


class SchedulerAdapter:
    """
    This is an adapter, adapts the interface of allm the different scheduler to the common
    scheduler interface (step). This is achieved by wrapping each scheduler by its
    corresponding subclass.
    """

    def step(self, logs, epoch, idx, iterations):
        raise NotImplementedError()


class ReduceLrOnPlateauAdapter(SchedulerAdapter):
    """
    Adapts the pytorch plateau-based LR scheduler.
    """

    __scheduler: ReduceLROnPlateau

    def __init__(self, scheduler: ReduceLROnPlateau):
        self.__scheduler = scheduler

    def step(self, logs, epoch, idx, iterations):
        self.__scheduler.step(logs["D_loss"])


class LrThresholdSchedulerAdapter(SchedulerAdapter):
    """
    Adapts the threshold-based LR scheduler.
    """

    __scheduler: LrThresholdScheduler

    def __init__(self, scheduler: LrThresholdScheduler):
        self.__scheduler = scheduler

    def step(self, logs, epoch, idx, iterations):
        self.__scheduler.step(logs["D_loss"], logs["GP"])


class CosineAnnealingWarmRestartsAdapter(SchedulerAdapter):
    """
    Adapts the pytorch LR scheduler cosine annealing with warm restart.
    """

    __scheduler: CosineAnnealingWarmRestarts

    def __init__(self, scheduler: CosineAnnealingWarmRestarts):
        self.__scheduler = scheduler

    def step(self, logs, epoch, idx, iterations):
        self.__scheduler.step(epoch + idx / iterations)


class MultiStepLrAdapter(SchedulerAdapter):
    """
    Adapts the pytorch LR scheduler cosine annealing with warm restart.
    """

    __scheduler: MultiStepLR

    def __init__(self, scheduler: MultiStepLR):
        self.__scheduler = scheduler

    def step(self, logs, epoch, idx, iterations):
        self.__scheduler.step()
