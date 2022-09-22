class ParameterClipper(object):
    """Clips weights of the parameters after each optim update.

    Ex: loss.backward()
        optimizer.step()
        model.apply(clipper)
    """

    __clip_min: int
    __clip_max: int

    def __init__(self, clip_min: int, clip_max: int):
        self.__clip_min = clip_min
        self.__clip_max = clip_max

    def __call__(self, module):
        if hasattr(module, "__rho"):
            weights = module.__rho.data
            weights = weights.clamp(self.__clip_max, self.__clip_max)
            module.__rho.data = weights
