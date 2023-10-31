import numpy as np
import torch
from algorithmn.cka import CKA
from algorithmn.fedgmm import FedGMMClient
from algorithmn.fedsr import FedSRClient
from data_loader import get_dataloaders, get_model, get_models
from datasets import RotatedMNIST
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.generator import Generator
from options import parse_args
from torch import nn

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
    args = parse_args()
    train_loaders, test_loaders = get_dataloaders(args)
    stu, ta, te = get_models(args)
    img, _ = next(iter(train_loaders[0]))
    x, _ = stu(img)
    y, _ = te(img)
    t, _ = ta(img)

    protos = [x, y, t]
    d = cal_protos_diff_vector(protos, y)
    v = optimize_collaborate_vector(d, 0.5, [0.5, 0.5, 0.5])
    cka = CKA(device="cpu")
    c = cka.linear_CKA(x, y)
    print(x.grad)
