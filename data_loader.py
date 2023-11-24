from argparse import Namespace
import json
import os
import random
from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from algorithmn.transform import DoubleTransform
from datasets import PACS, MultipleDomainDataset, RotatedMNIST
from models.base import FedModel
from pytorch_cinic.dataset import CINIC10
from models.cnn import CNN_FMNIST, MNISTCNN, CifarCNN, CifarCNN2
from models.mlp import FMNISTMLP, MNISTMLP, CifarMLP
from models.resnet import (
    CifarResNet,
    EMNISTResNet,
    FMNISTResNet,
    MNISTResNet,
    PACSResNet,
    RMNISTResNet,
)

DATASET_PATH = "./data"

AUGMENT_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
        transforms.ToTensor(),
    ]
)


class DatasetSplit(Dataset):
    def __init__(self, dataset, index=None, get_index=False, attack=False):
        super().__init__()
        self.get_index = get_index
        self.dataset = dataset
        self.attack = attack
        self.idxs = (
            [int(i) for i in index]
            if index is not None
            else [i for i in range(len(dataset))]
        )
        self.augment_set = []

    def do_augment(self, labels: List[int], aug_num: List[int], aug_transform):
        aug_cnt = {label: 0 for label in labels}
        label_aug_num_map = {label: aug_num[idx] for (idx, label) in enumerate(labels)}
        rest_label_set = set(
            [label for (idx, label) in enumerate(labels) if aug_num[idx] > 0]
        )

        for idx in self.idxs:
            x, label = self.dataset[idx]
            if label not in rest_label_set:
                continue

            x = aug_transform(x)
            self.augment_set.append((x, label))
            aug_cnt[label] += 1
            if aug_cnt[label] == label_aug_num_map[label]:
                rest_label_set.remove(label)
                if len(rest_label_set) == 0:
                    return

    def __len__(self):
        return len(self.idxs) + len(self.augment_set)

    def __getitem__(self, index):
        if index < len(self.idxs):
            x, label = self.dataset[self.idxs[index]]
        else:
            x, label = self.augment_set[index - len(self.idxs)]

        if self.attack:
            label = random.randint(0, label)
        if self.get_index:
            return x, label, index
        return x, label


def gen_data_loaders(
    dataset: Dataset,
    client_idxs: List[List[int]],
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    if shuffle:
        with open("./train_cfg/beta_2.0/cinic10_train_client_20_dirichlet.json", "w") as f:
            f.write(json.dumps(client_idxs))
    else:
        with open("./train_cfg/beta_2.0/cinic10_test_client_20_dirichlet.json", "w") as f:
            f.write(json.dumps(client_idxs))
    dataloaders = []
    for client_idx in client_idxs:
        splitted_dataset = DatasetSplit(dataset, list(client_idx), get_index)
        dataloader = DataLoader(
            dataset=splitted_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        dataloaders.append(dataloader)
    return dataloaders


def mnist_iid(
    dataset: datasets.MNIST,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
) -> Tuple[List[DataLoader], List[List[int]]]:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: MNIST
    :param num_clients: number of clients
    :return: list of dataloader
    """
    num_items = int(len(dataset) / num_clients)
    all_idxs = [i for i in range(len(dataset))]
    client_idxs = []
    for _ in range(num_clients):
        select_set = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        client_idxs.append(select_set)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def iid_partition(
    dataset: datasets.VisionDataset,
    num_clients: int,
):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """
    num_classes = len(np.unique(dataset.targets))
    shard_per_user = num_classes
    imgs_per_shard = int(len(dataset) / (num_clients * shard_per_user))
    client_idxs = [np.array([], dtype="int64") for _ in range(num_clients)]
    idxs_dict = {}
    for i in range(len(dataset)):
        label = dataset.targets[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    rand_set_all = []
    if len(rand_set_all) == 0:
        for i in range(num_clients):
            x = np.random.choice(np.arange(num_classes), shard_per_user, replace=False)
            rand_set_all.append(x)

    # divide and assign
    for i in range(num_clients):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            x = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_set.append(x)
        client_idxs[i] = np.concatenate(rand_set)

    for value in client_idxs:
        assert (len(np.unique(torch.tensor(dataset.targets)[value]))) == shard_per_user

    return client_idxs


def cifar10_iid(
    dataset: datasets.CIFAR10,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    client_idxs = iid_partition(dataset, num_clients)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def cinic10_iid(
    dataset: CINIC10,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    client_idxs = iid_partition(dataset, num_clients)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def fmnist_iid(
    dataset: datasets.CIFAR100,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    client_idxs = iid_partition(dataset, num_clients)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def emnist_iid(
    dataset: datasets.EMNIST,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    client_idxs = iid_partition(dataset, num_clients)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def cifar100_iid(
    dataset: datasets.CIFAR100,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    client_idxs = iid_partition(dataset, num_clients)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def cifar10_noniid(
    dataset: datasets.CIFAR10,
    num_clients: int,
    noniid_percent: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
    local_size=600,
    train=True,
):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    num_per_client = local_size if train else 300
    num_classes = len(np.unique(dataset.targets))

    noniid_labels_list = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]]

    # -------------------------------------------------------
    # divide the first dataset
    num_imgs_noniid = int(num_per_client * noniid_percent)
    num_imgs_iid = num_per_client - num_imgs_iid
    client_idxs = [np.array([]) for _ in range(num_clients)]
    num_samples = len(dataset)
    num_per_label_total = int(num_samples / num_classes)
    labels1 = np.array(dataset.targets)
    idxs1 = np.arange(len(dataset.targets))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = [i for i in range(num_classes)]
    # number of imgs has allocated per label
    label_used = (
        [2000 for _ in range(num_classes)]
        if train
        else [500 for _ in range(num_classes)]
    )
    iid_per_label = int(num_imgs_iid / num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes - 1) * iid_per_label

    for i in range(num_clients):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y * num_per_label_total + label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y] + iid_num) > num_per_label_total:
                start = y * num_per_label_total
                label_used[y] = 0
            client_idxs[i] = np.concatenate(
                (client_idxs[i], idxs[start : start + iid_num]), axis=0
            )
            label_used[y] = label_used[y] + iid_num

        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = noniid_labels_list[i % 5]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid / noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num * (noniid_labels - 1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y * num_per_label_total + label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y] + noniid_num) > num_per_label_total:
                start = y * num_per_label_total
                label_used[y] = 0
            client_idxs[i] = np.concatenate(
                (client_idxs[i], idxs[start : start + noniid_num]), axis=0
            )
            label_used[y] = label_used[y] + noniid_num
        client_idxs[i] = client_idxs[i].astype(int)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def dirichlet_partition(
    num_dataset: int, num_clients: int, targets: np.array, beta: float
) -> List[List[int]]:
    min_size = 0
    min_require_size = 10
    num_classes = len(np.unique(targets))
    while min_size < min_require_size:
        client_idxs = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)

            # generate dirichlet distribution: a possibility distribution over all clients for a class k
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            # if exceed the max num (namely num_dataset / num_clients), drop it
            proportions = np.array(
                [
                    p * (len(idx_j) < num_dataset / num_clients)
                    for p, idx_j in zip(proportions, client_idxs)
                ]
            )
            # normalize
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_idxs = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(client_idxs, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in client_idxs])

    for _ in range(num_clients):
        np.random.shuffle(client_idxs)
    return client_idxs


def cifar10_noniid_dirichlet(
    dataset: datasets.CIFAR10,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = np.array(dataset.targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def cinic10_noniid_dirichlet(
    dataset: CINIC10,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = []
    for _, label in dataset:
        targets.append(label)
    targets = np.array(targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def cifar100_noniid_dirichlet(
    dataset: datasets.CIFAR100,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = np.array(dataset.targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def fmnist_noniid_dirichlet(
    dataset: datasets.FashionMNIST,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = np.array(dataset.targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def mnist_noniid_dirichlet(
    dataset: datasets.MNIST,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = np.array(dataset.targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def emnist_noniid_dirichlet(
    dataset: datasets.EMNIST,
    num_clients: int,
    beta: float,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
):
    num_dataset = len(dataset)
    targets = np.array(dataset.targets)
    client_idxs = dirichlet_partition(num_dataset, num_clients, targets, beta)
    return gen_data_loaders(dataset, client_idxs, batch_size, shuffle, get_index)


def fmnist_dataset(double_trans=False) -> Tuple[Dataset, Dataset]:
    train_transform = test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if double_trans:
        crop_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=28, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                train_transform,
            ]
        )
        train_transform = DoubleTransform(crop_transforms)
    trainset = datasets.FashionMNIST(
        "data",
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = datasets.FashionMNIST(
        "data",
        train=False,
        transform=test_transform,
    )
    return trainset, testset


def emnist_dataset() -> Tuple[Dataset, Dataset]:
    trainset = datasets.EMNIST(
        "data",
        split="letters",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    testset = datasets.EMNIST(
        "data",
        split="letters",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    return trainset, testset


def mnist_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = datasets.MNIST(
        DATASET_PATH, train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(DATASET_PATH, train=False, transform=transform)
    return trainset, testset


def cinic10_dataset() -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )

    root = os.path.join(DATASET_PATH, "cinic10")
    trainset = CINIC10(
        root, partition="train", download=True, transform=transform_train
    )
    testset = CINIC10(root, partition="test", download=True, transform=transform_test)
    return trainset, testset


def cifar10_dataset(double_trans=False) -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if double_trans:
        transform_train = DoubleTransform(transform_train)

    trainset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )
    return trainset, testset


def cifar100_dataset() -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR100(
        root="data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR100(
        root="data", train=False, download=True, transform=transform_test
    )
    return trainset, testset


def get_dg_dataloaders(
    dataset: MultipleDomainDataset, test_env, num_clients, batch_size, get_index
):
    train_env = list(range(len(dataset.ENVIRONMENTS)))
    train_env.remove(test_env)
    train_loaders, test_loaders = [None] * num_clients, []
    num_train_env = len(train_env)

    train_client_idxs = [[] for _ in range(num_clients)]
    test_client_idxs = [[] for _ in range(num_clients)]

    env_num_clients_map = {}
    for i in range(num_train_env):
        env_num_clients_map[i] = num_clients // num_train_env

    for i in range(num_clients % num_train_env):
        env_num_clients_map[i] += 1

    for i in range(num_train_env):
        env = train_env[i]
        train_dataset = dataset[env]

        dataset_len = len(dataset[env])
        all_idxs = list(range(dataset_len))

        env_num_clients = env_num_clients_map[i]
        data_size_per_client = dataset_len // env_num_clients
        for j in range(env_num_clients):
            idx = i + j * num_train_env
            select_set = set(
                np.random.choice(all_idxs, data_size_per_client, replace=False)
            )
            all_idxs = list(set(all_idxs) - select_set)
            client_idxs = list(map(int, list(select_set)))
            train_client_idxs[idx] = client_idxs
            splitted_dataset = DatasetSplit(
                dataset=train_dataset, index=client_idxs, get_index=get_index
            )
            train_loaders[idx] = DataLoader(
                dataset=splitted_dataset, batch_size=batch_size, shuffle=True
            )

    test_dataset = DatasetSplit(dataset=dataset[test_env], get_index=get_index)

    for idx in range(num_clients):
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        test_loaders.append(test_loader)
        test_client_idxs[idx] = list(range(len(test_dataset)))

    return train_loaders, test_loaders


def pacs(
    num_clients: int, batch_size: int, get_index: bool, test_env=0
) -> Tuple[List[DataLoader], List[DataLoader]]:
    pacs_dataset = PACS(root=DATASET_PATH, test_envs=[test_env], augment=False)
    return get_dg_dataloaders(
        dataset=pacs_dataset,
        test_env=test_env,
        num_clients=num_clients,
        batch_size=batch_size,
        get_index=get_index,
    )


def rmnist(
    num_clients: int, batch_size: int, get_index: bool, test_env=0
) -> Tuple[List[DataLoader], List[DataLoader]]:
    rmnist_dataset = RotatedMNIST(
        root=DATASET_PATH, test_envs=[test_env], augment=False
    )
    return get_dg_dataloaders(
        dataset=rmnist_dataset,
        test_env=test_env,
        num_clients=num_clients,
        batch_size=batch_size,
        get_index=get_index,
    )


def read_client_idxs_from_json(json_path: str):
    with open(json_path, "r") as f:
        client_idxs = json.loads(f.read())
        return client_idxs


def get_dataloaders_from_json(
    args: Namespace, train_json_path: str, test_json_path: str
) -> Tuple[List[DataLoader], List[DataLoader]]:
    dataset = args.dataset
    local_bs = args.local_bs
    get_index = args.get_index
    double_trans = args.train_rule == "FedClassAvg"
    if dataset in ["cifar", "cifar10"]:
        trainset, testset = cifar10_dataset(double_trans)
    elif dataset in ["cinic10", "cinic"]:
        trainset, testset = cinic10_dataset()
    elif dataset == "cifar100":
        trainset, testset = cifar100_dataset()
    elif dataset == "mnist":
        trainset, testset = mnist_dataset()
    elif dataset == "fmnist":
        trainset, testset = fmnist_dataset(double_trans)
    elif dataset == "emnist":
        trainset, testset = emnist_dataset()
    else:
        raise NotImplementedError()
    train_loaders = gen_data_loaders(
        trainset,
        read_client_idxs_from_json(train_json_path),
        local_bs,
        True,
        get_index,
    )

    test_loaders = gen_data_loaders(
        testset,
        read_client_idxs_from_json(test_json_path),
        local_bs,
        False,
        get_index,
    )

    return train_loaders, test_loaders


# get_dataloaders returns train and test dataloader of given dataset
def get_dataloaders(args: Namespace) -> Tuple[List[DataLoader], List[DataLoader]]:
    dataset = args.dataset
    iid = args.iid
    num_clients = args.num_clients
    local_bs = args.local_bs
    get_index = args.get_index

    train_loaders, test_loaders = [], []
    if dataset == "mnist":
        trainset, testset = mnist_dataset()
        if iid:
            train_loaders = mnist_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = mnist_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = mnist_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = mnist_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["cifar10", "cifar"]:
        trainset, testset = cifar10_dataset()
        if iid:
            train_loaders = cifar10_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = cifar10_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = cifar10_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = cifar10_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["cinic", "cinic10"]:
        trainset, testset = cinic10_dataset()
        if iid:
            train_loaders = cinic10_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = cinic10_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = cinic10_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = cinic10_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["fmnist"]:
        trainset, testset = fmnist_dataset()
        if iid:
            train_loaders = fmnist_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = fmnist_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = fmnist_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = fmnist_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["emnist"]:
        trainset, testset = emnist_dataset()
        if iid:
            train_loaders = emnist_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = emnist_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = emnist_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = emnist_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["cifar100"]:
        trainset, testset = cifar100_dataset()
        if iid:
            train_loaders = cifar100_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders = cifar100_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders = cifar100_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders = cifar100_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["pacs"]:
        return pacs(num_clients=num_clients, batch_size=local_bs, get_index=get_index)
    elif dataset in ["rmnist"]:
        return rmnist(num_clients=num_clients, batch_size=local_bs, get_index=get_index)
    else:
        raise NotImplementedError()

    return train_loaders, test_loaders


def get_model(args: Namespace) -> nn.Module:
    dataset = args.dataset.lower()
    device = args.device
    num_classes = args.num_classes
    model_het = args.model_het
    prob = args.prob
    z_dim = args.z_dim
    if dataset in ["cifar", "cifar10", "cinic", "cinic_sep"]:
        global_model = CifarCNN(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        args.lr = 0.02
    elif dataset in ["pacs"]:
        global_model = PACSResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    elif dataset == "rmnist":
        global_model = RMNISTResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    elif dataset == "fmnist":
        global_model = CNN_FMNIST()
    elif dataset == "emnist":
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=num_classes)
    else:
        raise NotImplementedError()
    return global_model.to(device)


def get_heterogeneous_model(args: Namespace) -> nn.Module:
    dataset = args.dataset
    device = args.device
    num_classes = args.num_classes
    model_het = args.model_het
    prob = args.prob
    z_dim = args.z_dim
    if dataset in ["cifar", "cifar10", "cinic", "cinic_sep"]:
        heterogeneous_model = CifarResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        ).to(device)
        args.lr = 0.02
    else:
        raise NotImplementedError()
    return heterogeneous_model


# get_models returns (student model, ta model, and teacher model)
def get_models(args: Namespace) -> Tuple[FedModel, FedModel, FedModel]:
    dataset = args.dataset
    num_classes = args.num_classes
    prob = args.prob
    z_dim = args.z_dim
    model_het = args.model_het
    student, ta, teacher = None, None, None

    if dataset == "mnist":
        student = MNISTMLP(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        ta = MNISTCNN(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        teacher = MNISTResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    elif dataset == "fmnist":
        student = FMNISTMLP(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        ta = MNISTCNN(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        teacher = FMNISTResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    elif dataset == "emnist":
        student = FMNISTMLP(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        ta = MNISTCNN(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        teacher = EMNISTResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    elif dataset in ["cifar", "cifar10", "cifar100", "cinic10"]:
        student = CifarMLP(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        ta = CifarCNN(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
        teacher = CifarResNet(
            num_classes=num_classes,
            probabilistic=prob,
            model_het=model_het,
            z_dim=z_dim,
        )
    else:
        raise NotImplementedError()
    return student, ta, teacher
