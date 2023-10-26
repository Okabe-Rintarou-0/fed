import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from data_loader import cifar10_dataset

from datasets import PACS
from models.cnn import CifarCNN
from models.resnet import PACSResNet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="pacs")
parser.add_argument("--weights", type=str)
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--out", default="./out")
args = parser.parse_args()

CONFIG_MAP = {
    "pacs": {
        "model": PACSResNet,
        "model_args": {"probabilistic": True, "num_classes": 7},
        "classes": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
    },
    "cifar": {
        "model": CifarCNN,
        "model_args": {"probabilistic": True, "num_classes": 10},
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    },
}


def collect_data(dataloader, model):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        z, output, (z_mu, z_sigma) = model(images, return_dist=True)
        predict = torch.argmax(output, dim=1)
        for i in range(len(predict)):
            p = predict[i].item()
            label_dist[p]["z"].append(z[i].cpu().detach().tolist())
            this_z_mu = z_mu[i] * r_C
            this_z_sigma = z_sigma[i] * r_C
            label_dist[p]["mu"].append(this_z_mu.mean().item())
            label_dist[p]["sigma"].append(this_z_sigma.mean().item())


if __name__ == "__main__":
    dataset = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device
    out_path = args.out

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    ModelClass = CONFIG_MAP[dataset]["model"]
    model_args = CONFIG_MAP[dataset]["model_args"]
    classes = CONFIG_MAP[dataset]["classes"]
    model = ModelClass(**model_args)

    state_dict = torch.load(weights_path, map_location=device)
    r_mus = state_dict["r.mu"].to(device)
    r_sigmas = state_dict["r.sigma"].to(device)
    r_C = state_dict["r.C"].to(device)

    state_dict.pop("r.mu")
    state_dict.pop("r.sigma")
    state_dict.pop("r.C")
    model.load_state_dict(state_dict)
    label_dist = {i: {"mu": [], "sigma": [], "z": []} for i in range(len(classes))}
    
    if dataset == "pacs":
        dataset = PACS(root="./data", test_envs=[0])
        for env in range(len(dataset.ENVIRONMENTS)):
            this_dataset = dataset[env]
            dataloader = DataLoader(dataset=this_dataset, batch_size=batch_size)
            collect_data(dataloader, model)
    elif dataset == "cifar":
        train_dataset, test_dataset = cifar10_dataset()
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
        collect_data(train_dataloader, model)
        collect_data(test_dataloader, model)

    for label in label_dist:
        dist = label_dist[label]
        mus = np.array(dist["mu"])
        sigmas = np.array(dist["sigma"])
        r_mu = r_mus[label].mean().item()
        r_sigma = r_sigmas[label].mean().item()
        plt.clf()
        plt.scatter(mus, sigmas, label=classes[label], s=20)
        plt.axhline(y=r_sigma, linestyle="--", label="$r_\mu$")
        plt.axvline(x=r_mu, linestyle="--", label="$r_\sigma$")
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        cls = classes[label]
        plt.legend()
        plt.savefig(os.path.join(out_path, f"{cls}.png"))
