from torchvision.datasets import MNIST

from loader import mnist_iid

if __name__ == '__main__':
    dataset = MNIST("./datasets/mnist", download=True)
    client_datasets = mnist_iid(dataset, 10)
    print(len(client_datasets))
    for dataset in client_datasets:
        print(dataset)
