from argparse import Namespace
from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from datasets import PACS

from models.cnn import CNN_FMNIST, CifarCNN
from models.resnet import CifarResNet, PACSResNet

DATASET_PATH = "./data"


class DatasetSplit(Dataset):
    def __init__(self, dataset, index=None, get_index=False):
        super().__init__()
        self.get_index = get_index
        self.dataset = dataset
        self.idxs = (
            [int(i) for i in index]
            if index is not None
            else [i for i in range(len(dataset))]
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        x, label = self.dataset[self.idxs[index]]
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
    dataloaders = []
    for client_idx in client_idxs:
        splitted_dataset = DatasetSplit(dataset, list(client_idx), get_index)
        dataloader = DataLoader(
            dataset=splitted_dataset, batch_size=batch_size, shuffle=shuffle
        )
        dataloaders.append(dataloader)
    return dataloaders, client_idxs


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


def cifar10_iid(
    dataset: datasets.CIFAR10,
    num_clients: int,
    batch_size: int,
    shuffle: bool,
    get_index: bool,
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


def mnist_dataset() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = datasets.MNIST(
        DATASET_PATH, train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(DATASET_PATH, train=False, transform=transform)
    return trainset, testset


def cifar10_dataset() -> Tuple[Dataset, Dataset]:
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

    trainset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )
    return trainset, testset


def pacs(
    num_clients: int, batch_size: int, get_index: bool, test_env=0
) -> Tuple[List[DataLoader], List[DataLoader]]:
    pacs_dataset = PACS(root=DATASET_PATH, test_envs=[test_env], augment=False)
    train_env = list(range(len(pacs_dataset.ENVIRONMENTS)))
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
        train_dataset = pacs_dataset[env]

        dataset_len = len(pacs_dataset[env])
        all_idxs = list(range(dataset_len))
        env_num_clients = env_num_clients_map[i]
        data_size_per_client = dataset_len // env_num_clients
        for j in range(env_num_clients):
            idx = i + j * num_train_env
            select_set = set(
                np.random.choice(all_idxs, data_size_per_client, replace=False)
            )
            all_idxs = list(set(all_idxs) - select_set)
            train_client_idxs[idx] = list(select_set)
            splitted_dataset = DatasetSplit(
                dataset=train_dataset, index=list(select_set), get_index=get_index
            )
            train_loaders[idx] = DataLoader(
                dataset=splitted_dataset, batch_size=batch_size, shuffle=True
            )

    test_dataset = DatasetSplit(dataset=pacs_dataset[test_env], get_index=get_index)

    for idx in range(num_clients):
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        test_loaders.append(test_loader)
        test_client_idxs[idx] = list(range(len(test_dataset)))

    return train_loaders, test_loaders, train_client_idxs, test_client_idxs


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
            train_loaders, train_client_idxs = mnist_iid(trainset, num_clients, local_bs, shuffle=True)
            test_loaders, test_client_idxs = mnist_iid(testset, num_clients, local_bs, shuffle=False)
    elif dataset in ["cifar10", "cifar"]:
        trainset, testset = cifar10_dataset()
        if iid:
            train_loaders, train_client_idxs = cifar10_iid(
                trainset, num_clients, local_bs, shuffle=True, get_index=get_index
            )
            test_loaders, test_client_idxs = cifar10_iid(
                testset, num_clients, local_bs, shuffle=False, get_index=get_index
            )
        else:
            train_loaders, train_client_idxs = cifar10_noniid_dirichlet(
                trainset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=True,
                get_index=get_index,
            )
            test_loaders, test_client_idxs = cifar10_noniid_dirichlet(
                testset,
                num_clients,
                args.beta,
                local_bs,
                shuffle=False,
                get_index=get_index,
            )
    elif dataset in ["pacs"]:
        return pacs(num_clients=num_clients, batch_size=local_bs, get_index=get_index)
    else:
        raise NotImplementedError()

    return train_loaders, test_loaders, train_client_idxs, test_client_idxs


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
