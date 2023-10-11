import numpy as np
from torchvision.datasets import MNIST


def mnist_iid(dataset: MNIST, num_clients: int):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: MNIST
    :param num_clients: number of clients
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    np.random.seed(2023)
    for i in range(num_clients):
        select_set = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        dict_users[i] = list(select_set)
    return dict_users
