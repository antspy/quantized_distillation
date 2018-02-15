import torch

USE_CUDA = torch.cuda.is_available()
from .quant_functions import uniformQuantization, nonUniformQuantization, ScalingFunction, \
                        uniformQuantization_variable, nonUniformQuantization_variable

__all__ = ('uniformQuantization', 'ScalingFunction', 'nonUniformQuantization',
           'uniformQuantization_variable', 'nonUniformQuantization_variable')