import copy
from typing import Any, Dict, List

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def aggregate_weights(w: List[Dict[str, Any]], agg_weight) -> Dict[str, Any]:
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(agg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]
    return w_avg

def write_client_datasets(idx: int, writer: SummaryWriter, dataloader: DataLoader, train: bool):
    tag = f'client_{idx}_{"train" if train else "test"}_dataset'
    data_iter = iter(dataloader)
    imgs, _ = next(data_iter)
    writer.add_images(tag, imgs)

def show_img_batch(imgs):
    imgs = torchvision.utils.make_grid(imgs).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.show()
