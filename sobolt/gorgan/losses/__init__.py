from .adversarial import AdversarialGeneratorLoss, AdversarialDiscriminatorLoss
from .wasserstein import WassersteinCriticLoss, WassersteinGeneratorLoss
from .perceptual import PerceptualLoss
from .minimax import MinimaxDiscriminatorLoss, MinimaxGeneratorLoss
from .hue import HueLoss
from .total_variation import TotalVariationLoss
from .local_phase_coherence import LocalPhaseCoherenceLoss
from .least_squared_error import (
    LeastSquaredErrorDiscriminatorLoss,
    LeastSquaredErrorGeneratorLoss,
)
from .ssim import SsimLoss
from .composite_loss import CompositeLoss
