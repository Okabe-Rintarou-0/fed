import os
from torchvision import datasets, transforms

from loader import mnist_iid
from utils.image_utils import show_img_batch

if __name__ == '__main__':
    need_download = os.path.exists("./datasets/mnist")
    dataset = datasets.MNIST("./datasets/mnist", download=need_download,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307), (0.3081))
                             ]))
    client_dataloaders = mnist_iid(dataset, 10, 20, True)
    for dataloader in client_dataloaders:
        images, labels = next(iter(dataloader))
        show_img_batch(images)
