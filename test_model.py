import numpy as np
import torch
from algorithmn.fedgmm import FedGMMClient
from algorithmn.fedsr import FedSRClient
from data_loader import get_dataloaders, get_model
from datasets import RotatedMNIST
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

    # wm = {
    #     i: {
    #         "r.mu": torch.rand(10, 128),
    #         "r.sigma": torch.rand(10, 128) * (i**2 + 1),
    #     }
    #     for i in range(20)
    # }

    # seed = 520
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    # client_idxs = list(range(20))

    RotatedMNIST.prepare("./data")

    # sd2 = torch.load("./training_data/global.pth", map_location="cpu")
    # print(sd2["r.sigma"] - sd["r.sigma"].mean(dim=1))
    # print(sd2["r.mu"] - sd["r.mu"].mean(dim=0))
    # print(sd["r.C"] -sd[""])
    # args = parse_args()
    # train_loaders, test_loaders = get_dataloaders(args=args)
    # model = get_model(args=args)
    # cli = FedSRClient(
    #     idx=0,
    #     args=args,
    #     train_loader=train_loaders[0],
    #     test_loader=test_loaders[0],
    #     local_model=model,
    # )

    # cli.local_model.load_state_dict(torch.load("./training_data/global.pth", map_location="cpu"))
    # print(cli.local_test())
