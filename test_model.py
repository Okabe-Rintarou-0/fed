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
    a_stu_beta_2 = [
        46.96,
        47.16,
        46.79,
        46.7,
        47.41,
        46.27,
        46.72,
        47.46,
        46.69,
        46.47,
        45.25,
        47.30,
        45.82,
        45.29,
        45.01,
        44.86,
        43.72,
        43.42,
        38.09,
        35.28,
    ]
    a_beta_2 = [
        46.96,
        48.09,
        49.21,
        50.21,
        52.6,
        53.74,
        55.61,
        57.98,
        58.79,
        60.88,
        61.72,
        64.08,
        65.73,
        67.20,
        69.13,
        70.90,
        72.18,
        74.11,
        75.31,
        76.99,
    ]

    x = np.array(range(0, 20))
    A = np.array(a_beta_2)
    A_stu = np.array(a_stu_beta_2)
    n = np.array([20] * 20)
    print((n * A - (n - x) * A_stu) / x)
