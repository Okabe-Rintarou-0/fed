import numpy as np
import torch
from algorithmn.cka import CKA
from algorithmn.fedgmm import FedGMMClient
from algorithmn.fedsr import FedSRClient
from data_loader import get_dataloaders, get_model, get_models
from datasets import RotatedMNIST
from torchvision.datasets import EMNIST
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.generator import Generator
from models.resnet import CifarResNet
from options import parse_args
from torch import nn

from sklearn.decomposition import KernelPCA

import matplotlib.pyplot as plt

from tools import (
    cal_dist_avg_difference_vector,
    cal_protos_diff_vector,
    optimize_collaborate_vector,
)


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn(3))


import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import linear_kernel


if __name__ == "__main__":
    d = EMNIST(root="data", download=True, split='byclass')
    print(d.targets)