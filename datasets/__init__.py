import os
import sys
_currDir = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_FOLDER = os.path.join(_currDir, 'saved_datasets')
PATH_PERL_SCRIPTS_FOLDER = os.path.abspath(os.path.join(_currDir, '..', 'perl_scripts'))

try:
    os.mkdir(BASE_DATA_FOLDER)
except:pass

from .CIFAR10 import CIFAR10
from .CIFAR100 import CIFAR100
from .PennTreeBank import PennTreeBank
from .ImageNet12 import ImageNet12
from .translation_datasets import multi30k_DE_EN, onmt_integ_dataset, WMT13_DE_EN
from .MNIST import MNIST
from .customs_datasets import LoadingTensorsDataset

__all__ = ('CIFAR10', 'PennTreeBank', 'WMT13_DE_EN', 'ImageNet12', 'multi30k_DE_EN',
           'onmt_integ_dataset', 'CIFAR100', 'MNIST', 'LoadingTensorsDataset')