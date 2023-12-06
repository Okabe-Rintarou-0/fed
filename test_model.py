import numpy as np
import torch
from algorithmn.fedgmm import FedGMMClient
from algorithmn.fedsr import FedSRClient
from data_loader import get_dataloaders, get_model, get_models
from datasets import RotatedMNIST
from torchvision.datasets import EMNIST
from models.cnn import CifarCNN
from models.generator import Generator
from models.mlp import FMNISTMLP, MNISTMLP, CifarMLP
from models.resnet import CifarResNet, FMNISTResNet, MNISTResNet
from options import parse_args
from torch import nn

from sklearn.decomposition import KernelPCA

import matplotlib.pyplot as plt

from tools import (
    weight_flatten,
)


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn(3))


import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import linear_kernel


if __name__ == "__main__":
    a_stu = [
        44.55,
        44.86,
        44.52,
        43.54,
        41.53,
        39.83,
        40.02,
        40.96,
        42.43,
        42.44,
        41.18,
        39.18,
        36.65,
        32.62,
        32.83,
        35.87,
        36.75,
        32.40,
        22.59,
        12.95,
    ]
    a = [
        44.55,
        46.21,
        46.63,
        48.03,
        47.94,
        48.29,
        49.93,
        51.41,
        54.79,
        55.75,
        57.96,
        58.56,
        59.99,
        61.33,
        63.86,
        67.43,
        69.12,
        71.00,
        71.42,
        73.89,
    ]

    x = np.array(range(0, 20))
    A = np.array(a)
    A_stu = np.array(a_stu)
    n = np.array([20] * 20)
    print((n * A - (n - x) * A_stu) / x)
