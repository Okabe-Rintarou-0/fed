from argparse import Namespace
from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn

from models.cnn import CNN_FMNIST, CifarCNN

DATASET_PATH = './data'


class DatasetSplit(Dataset):
    def __init__(self, dataset, index=None):
        super().__init__()
        self.dataset = dataset
        self.idxs = [int(i) for i in index] if index is not None else [
            i for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x, label = self.dataset[self.idxs[item]]
        return x, label


def mnist_iid(dataset: datasets.MNIST, num_clients: int, batch_size: int, shuffle: bool) -> List[DataLoader]:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: MNIST
    :param num_clients: number of clients
    :return: list of dataloader
    """
    num_items = int(len(dataset) / num_clients)
    dataloaders, all_idxs = [], [i for i in range(len(dataset))]
    for _ in range(num_clients):
        select_set = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        splitted_dataset = DatasetSplit(dataset, list(select_set))
        dataloader = DataLoader(
            dataset=splitted_dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)
    return dataloaders


def cifar10_iid(dataset: datasets.CIFAR10, num_clients: int, batch_size: int, shuffle: bool):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """
    num_classes = len(np.unique(dataset.targets))
    shard_per_user = num_classes
    imgs_per_shard = int(len(dataset) / (num_clients * shard_per_user))
    client_idxs = [np.array([], dtype='int64') for _ in range(num_clients)]
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_clients):
            x = np.random.choice(np.arange(num_classes),
                                 shard_per_user, replace=False)
            rand_set_all.append(x)

    dataloaders = []
    # divide and assign
    for i in range(num_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            x = np.random.choice(
                idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        client_idxs[i] = np.concatenate(rand_set)
        splitted_dataset = DatasetSplit(dataset, list(client_idxs[i]))
        dataloader = DataLoader(
            dataset=splitted_dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)

    for value in client_idxs:
        assert(
            len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return dataloaders


def mnist_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(DATASET_PATH, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(DATASET_PATH, train=False, transform=transform)
    return trainset, testset

def cifar10_dataset() -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    return trainset, testset


# get_dataloaders returns train and test dataloader of given dataset
def get_dataloaders(args: Namespace) -> Tuple[List[DataLoader], List[DataLoader]]:
    dataset = args.dataset
    iid = args.iid
    num_clients = args.num_clients
    local_bs = args.local_bs

    train_loaders, test_loaders = [], []
    if dataset == 'mnist':
        trainset, testset = mnist_dataset()
        if iid:
            train_loaders = mnist_iid(
                trainset, num_clients, local_bs, shuffle=True)
            test_loaders = mnist_iid(
                testset, num_clients, local_bs, shuffle=False)
    elif dataset in ['cifar10', 'cifar']:
        trainset, testset = cifar10_dataset()
        if iid:
            train_loaders = cifar10_iid(
                trainset, num_clients, local_bs, shuffle=True
            )
            test_loaders = cifar10_iid(
                testset, num_clients, local_bs, shuffle=False
            )
    else:
        raise NotImplementedError()

    return train_loaders, test_loaders

def get_model(args: Namespace) -> nn.Module:
    dataset = args.dataset
    device = args.device
    if dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
        global_model = CifarCNN(num_classes=args.num_classes).to(device)
        args.lr = 0.02
    elif dataset == 'fmnist':
        global_model = CNN_FMNIST().to(device)
    elif dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
    else:
        raise NotImplementedError()
