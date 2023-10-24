import torch
from algorithmn.fedgmm import FedGMMClient
from data_loader import get_dataloaders
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.generator import Generator
from options import parse_args
from torch import nn


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn(3))


if __name__ == "__main__":
    g = Generator(num_classes=10, z_dim=128)
    print(g(torch.tensor([[1], [2]])))
    # x, y = resnet(torch.zeros(3, 3, 32, 32))
    # print(y.size())
    # print(torch.sum(y, 1))
