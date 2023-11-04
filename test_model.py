import numpy as np
import torch
from algorithmn.cka import CKA
from algorithmn.fedgmm import FedGMMClient
from algorithmn.fedsr import FedSRClient
from data_loader import get_dataloaders, get_model, get_models
from datasets import RotatedMNIST
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
    # args = parse_args()
    # train_loaders, test_loaders = get_dataloaders(args)
    # stu, ta, te = get_models(args)
    # img, _ = next(iter(train_loaders[0]))
    # x, _ = stu(img)
    # y, _ = te(img)
    # t, _ = ta(img)

    # protos = [x, y, t]
    # d = cal_protos_diff_vector(protos, y)
    # v = optimize_collaborate_vector(d, 0.5, [0.5, 0.5, 0.5])
    # cka = CKA(device="cpu")
    # c = cka.linear_CKA(torch.randn((2, 128)), torch.randn((2, 128)) * 0.0000015)
    # print(c)
    # m = CifarResNet(num_classes=10)
    # print(m)
    # a = torch.ones(1, 128)
    # b = torch.ones(1, 128) * 0
    # c = torch.ones(1, 128) * (1 - 1e-20)
    # pca = KernelPCA(n_components=2, gamma=1)
    # pca = pca.fit_transform(torch.vstack([a, b, c]))
    # print(pca)
    # x = pca[:, 0]
    # y = pca[:, 1]

    # # 绘制散点图
    # plt.scatter(x, y, color="b", marker="o", label="Data Points")

    # # 设置图表标题和轴标签
    # plt.title("N×2 Matrix Plot")
    # plt.xlabel("X Axis")
    # plt.ylabel("Y Axis")
    # plt.show()

    dv = [
        -0.7474,
        -0.7439,
        -0.8306,
        -0.6384,
        -0.7246,
        -0.8268,
        -0.7689,
        -0.5976,
        -0.9209,
        -0.7316,
        -0.6427,
        -0.7920,
        -0.9127,
        -0.6835,
        -0.6927,
        -0.6472,
        -0.63348,
        -0.8200,
        -0.6470,
        -0.7392,
    ]

    agg = optimize_collaborate_vector(torch.tensor(dv), 0.3, torch.ones(20) / 20)
    print(agg)

    y = np.random.choice(list(range(10)), 32)
    y2 = np.random.choice(list(range(10)), 32)
    y_input = F.one_hot(torch.LongTensor(y), 10)
    y_input2 = F.one_hot(torch.LongTensor(y2), 10)
    print(y_input.shape, y_input2.shape)
    lam = torch.rand(32, 1)
    mixup = lam * y_input + (1 - lam) * y_input2
    print(mixup)

    print(np.array([1, 2]) * np.array([1, 2]))
