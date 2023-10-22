import torch
from algorithmn.fedgmm import FedGMMClient
from data_loader import get_dataloaders
from models.cnn import PACSCNN, CifarCNN, ComplexCNN
from models.resnet import CifarResnet
from options import parse_args
from torch import nn


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Parameter(torch.randn(3))


if __name__ == "__main__":
    # test = TestModule()
    # cnn.add_module('test', test)
    # print(cnn.test.C)
    # print(test.C)

    # x = torch.sum(test.C ** 2)
    # x.backward()

    # print(cnn.test.C.grad)
    # print(test.C.grad)

    # x = torch.sum(test.C ** 2)
    # x.backward()

    # print(cnn.test.C.grad)
    # print(test.C.grad)
    # for key, _ in cnn.named_parameters():
    #     print(key)
    # x, y = cnn(torch.zeros(1, 3, 32, 32))
    # print(y.size())
    # print(torch.sum(y, 1))

    cnn = ComplexCNN()
    print(cnn(torch.zeros(3, 3, 32, 32)))
    # x, y = resnet(torch.zeros(3, 3, 32, 32))
    # print(y.size())
    # print(torch.sum(y, 1))
