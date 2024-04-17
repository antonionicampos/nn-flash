from .critics import MLPCritic, DeepConvCritic
from .generators import MLPGenerator, DeepConvGenerator
from .utils import CustomHistory
from .wgan import (
    WGAN,
    WGANGP,
    critic_loss,
    generator_loss,
)

__all__ = [
    "MLPCritic",
    "MLPGenerator",
    "DeepConvCritic",
    "DeepConvGenerator",
    "WGAN",
    "WGANGP",
    "critic_loss",
    "generator_loss",
    "CustomHistory",
]
