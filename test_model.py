import torch
from models.cnn import CifarCNN
from models.resnet import CifarResnet


if __name__ == '__main__':
    cnn = CifarCNN(probabilistic=True)
    for key, _ in cnn.named_parameters():
        print(key)
    x, y = cnn(torch.zeros(1, 3, 32, 32))
    print(y.size())
    print(torch.sum(y, 1))

    resnet = CifarResnet(probabilistic=True)
    x, y = resnet(torch.zeros(3, 3, 32, 32))
    print(y.size())
    print(torch.sum(y, 1))