from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

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
from scipy.stats import norm, multivariate_normal

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


def collect_data(dataloader, model, label_dist):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        z, output, (z_mu, z_sigma) = model(images, return_dist=True)
        predict = torch.argmax(output, dim=1)
        for i in range(len(predict)):
            p = predict[i].item()
            label_dist[p]["z"].append(z[i].cpu().detach().tolist())


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
    label_dist = {
        i: {"mu": [], "sigma": [], "z": [], "wrong_mu": [], "wrong_sigma": []}
        for i in range(len(classes))
    }
    pca = PCA(n_components=2)
    if dataset == "pacs":
        for env in range(len(dataset.ENVIRONMENTS)):
            this_dataset = dataset[env]
            dataloader = DataLoader(dataset=this_dataset, batch_size=batch_size)
            collect_data(dataloader, model, label_dist)
    elif dataset == "cifar":
        train_dataset, _ = cifar10_dataset()
        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        collect_data(dataloader, model, label_dist)

    zlist = []
    labels_cnt = []

    for label in label_dist:
        dist = label_dist[label]
        zs = np.array(dist["z"])
        if len(zs) == 0:
            print(label)
            continue
        zlist.extend(zs.tolist())
        labels_cnt.append(len(zs))
        zs = pca.fit_transform(zs)
        r_mu = r_mus[label].cpu().detach().numpy()
        r_sigma = r_sigmas[label].cpu().detach().numpy()
        samples = np.random.multivariate_normal(r_mu, np.diag(r_sigma), 1000)
        samples = pca.fit_transform(samples)
        r_mu = np.mean(samples, axis=0)
        r_sigma = np.cov(samples, rowvar=False)
        plt.clf()
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            alpha=0.2,
            label="Samples of $r(z|y)$",
            s=10,
        )
        plt.scatter(zs[:, 0], zs[:, 1], alpha=0.2, label="Samples of $p(z|y)$", s=10)
        cls = classes[label]
        plt.legend()
        plt.savefig(os.path.join(out_path, f"{cls}.png"))

    zlist = np.array(zlist)
    labels_cnt = np.array(labels_cnt)
    zlist = pca.fit_transform(zlist)

    start = 0
    end = 0
    plt.clf()
    for label in range(len(labels_cnt)):
        label_cnt = labels_cnt[label]
        end = start + label_cnt
        this_zs = zlist[start:end]
        start = end

        plt.scatter(this_zs[:, 0], this_zs[:, 1], alpha=0.2, label=classes[label], s=10)
    plt.legend()
    plt.savefig(os.path.join(out_path, f"dist.png"))
