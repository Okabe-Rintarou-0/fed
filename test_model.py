import numpy as np
import torch
from algorithmn.fedgmm import FedGMMClient
from data_loader import get_dataloaders
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.generator import Generator
from options import parse_args
from torch import nn

from tools import cal_dist_avg_difference_vector, optimize_collaborate_vector


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn(3))


if __name__ == "__main__":
    # g = Generator(num_classes=10, z_dim=128)
    # print(g(torch.tensor([[1], [2]])))
    # x, y = resnet(torch.zeros(3, 3, 32, 32))
    # print(y.size())
    # print(torch.sum(y, 1))

    wm = {
        i: {
            "r.mu": torch.rand(10, 128),
            "r.sigma": torch.rand(10, 128) * (i**2 + 1),
        }
        for i in range(20)
    }

    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    client_idxs = list(range(20))

    dv = cal_dist_avg_difference_vector(client_idxs, wm)
    print("dv", dv)
    cv = optimize_collaborate_vector(dv, 0.8, [0.1 for _ in range(20)])
    print("cv", cv)
