from typing import List, Dict, Optional

import torch.optim as optim
from torch._six import inf


class LrThresholdScheduler:
    """A scheduler for decreasing a nn.Module's optimizer learning rate by a
    specified factor if a threshold is reached on an independent function.

    Because we want to ensure gradients are not updated before decreasing the learning
    rate (as the threshold used also indicates the run is not performing as intended),
    the threshold is set outside of this function to have control over the optimizer step.
    Functions and variables name derived from Pytorch's ReduceonPLateau LR scheduler.
    """

    __optimizer: optim.Optimizer
    __min_lrs: List[float]
    __factor: float
    __factor_decay: float
    __best: float
    __eps: float
    __patience: int
    __cooldown: int
    __threshold: int
    __cooldown_counter: int
    __plateau_counter: int

    def __init__(
        self,
        optimizer: optim.Optimizer,
        factor: float = 0.5,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        patience: int = 10,
        cooldown: int = 0,
        threshold: int = 10000000,
        factor_decay: float = 1.0,
    ):
        """
        Initializes the LrThresholdScheduler class.

        Parameters
        ----------
        optimizer: optim.Optimizer
            An nn.Module's optimizer, currently we use ADAM.
        factor: float
            The factor we want to decrease the current LR with (default is 0.5)
        min_lr: float
            A scalar for the lower bound of the LR (default is 0).
        eps: float
            Following Pytorch's implementation, no decrease in LR if difference between
            old and new LR is smaller than eps.
        patience: int
            Number of iterations with no improvement after which we apply the decay
            scalar (factor_decay) to the factor.
        cooldown: int
            Iteration number to elapse after decreasing the LR before re-starting the
            scheduler.
        threshold: int
            The value of a function that allows for the LR to be decreased.
        factor_decay: float
            The scalar to decrease the LR decreaser factor with over iterations.
        """
        self.__optimizer = optimizer
        self.__min_lrs = [min_lr]

        if factor >= 1.0:
            raise ValueError("Factor ({}) needs to be less than 1.0".format(factor))
        self.__factor = factor
        self.__factor_decay = factor_decay
        self.__eps = eps
        self.__patience = patience
        self.__cooldown = cooldown
        self.__threshold = threshold
        self.__cooldown_counter = 0
        self.__best = 0
        self.__plateau_counter = 0

        self.__reset()

    def __reset(self):
        """
        Resets the counters.
        """
        self.__best = inf
        self.__cooldown_counter = 0
        self.__plateau_counter = 0

    def step(self, cooldown_function: float, function_to_threshold: float):
        """
        The process by which we take the LR parameter from an optimizer and decrease
        its learning rate.

        The function updates the optimizer LR according to a pre-specified factor. This
        allows us to dynamically modify an important parameter during a network training.

        Keeping track of iterations allows us to make sure the learning rate is not
        continuously decreased if the objective function value stays above the
        threshold for x iterations.

        Parameters
        ----------
        cooldown_function: float
        A function to determine a threshold LR decrease. D_loss is the default.
        function_to_threshold: float
            A function we want to compare to a threshold. Gradient penalty (GP) is the
            default.
        factor_decay: float
            The scalar we want to decrease the learning rate value with everytime a
            threshold is encountered.
        """
        rel_epsilon = 1.0 - 1e-4
        if cooldown_function < self.__best * rel_epsilon:
            self.__best = cooldown_function
            self.__plateau_counter = 0
        else:
            self.__plateau_counter += 1

        if self.__cooldown_counter > 0:
            self.__cooldown_counter -= 1
            self.__plateau_counter = 0

        if self.__plateau_counter > self.__patience:
            self.__factor = self.__factor * self.__factor_decay
            self.__decrease_lr(function_to_threshold)
            self.__cooldown_counter = self.__cooldown
            self.__plateau_counter = 0

    def __decrease_lr(self, function_to_threshold: float):
        """
        The function decreases the learning rate for the wrapped optimizer if a
        a function value threshold (gradient penalty in this case) is reached.

        Parameters
        ----------
        function_to_threshold: float
            A function we want to compare to a threshold.
        """
        for idx, param_group in enumerate(self.__optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.__factor, self.__min_lrs[idx])
            if function_to_threshold > self.__threshold:
                if old_lr - new_lr > self.__eps:
                    param_group["lr"] = new_lr
                    print("Decreasing LR from {} to {:.4e}".format(idx, new_lr))
