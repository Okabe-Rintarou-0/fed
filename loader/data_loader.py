from typing import List
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset


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


def mnist_iid(dataset: MNIST, num_clients: int, batch_size: int, shuffle: bool) -> List[DataLoader]:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: MNIST
    :param num_clients: number of clients
    :return: list of dataloader
    """
    num_items = int(len(dataset) / num_clients)
    dataloaders, all_idxs = [], [i for i in range(len(dataset))]
    np.random.seed(2023)
    for _ in range(num_clients):
        select_set = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - select_set)
        splitted_dataset = DatasetSplit(dataset, list(select_set))
        dataloader = DataLoader(
            dataset=splitted_dataset, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(dataloader)
    return dataloaders
