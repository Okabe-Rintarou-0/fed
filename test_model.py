import numpy as np
import torch
from algorithmn.fedgmm import FedGMMClient
from data_loader import get_dataloaders
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.generator import Generator
from options import parse_args
from torch import nn

from tools import cal_dist_avg_difference_vector, optimize_collabrate_vector


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

    mu1 = torch.rand((10, 128))
    sigma1 = torch.rand((10, 128))

    d1 = {
        'r.mu': mu1,
        'r.sigma': sigma1
    }

    mu2 = torch.rand((10, 128))
    sigma2 = torch.rand((10, 128))

    d2 = {
        'r.mu': mu2,
        'r.sigma': sigma2
    }

    wm = {
        0: d1,
        1: d2
    }

    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(d1)
    print(d2)
    client_idxs = [0, 1]

    dv = cal_dist_avg_difference_vector(client_idxs, wm)
    print('dv', dv)
    cv = torch.rand((2,))
    optimize_collabrate_vector(cv, client_idxs, dv, 0.1, [0.5, 0.5])
    print('cv', cv)