import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets import PACS
from models.resnet import PACSResNet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="pacs")
parser.add_argument("--weights", type=str)
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--out", default="./out")
args = parser.parse_args()

if __name__ == "__main__":
    dataset = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device
    out_path = args.out

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if dataset == "pacs":
        model = PACSResNet(num_classes=7, probabilistic=True).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        r_mus = state_dict["r.mu"].to(device)
        r_sigmas = state_dict["r.sigma"].to(device)
        r_C = state_dict["r.C"].to(device)
        state_dict.pop("r.mu")
        state_dict.pop("r.sigma")
        state_dict.pop("r.C")
        model.load_state_dict(state_dict)
        dataset = PACS(root="./data", test_envs=[0])
        label_dist = {i: {"mu": [], "sigma": [], "z": []} for i in range(7)}

        for env in range(len(dataset.ENVIRONMENTS)):
            this_dataset = dataset[env]
            dataloader = DataLoader(dataset=this_dataset, batch_size=batch_size)
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                z, output, (z_mu, z_sigma) = model(images, return_dist=True)
                predict = torch.argmax(output, dim=1)
                r_sigma = r_sigmas[labels]
                r_mu = r_mus[labels]
                for i in range(len(predict)):
                    p = predict[i].item()
                    label_dist[p]["z"].append(z[i].cpu().detach().tolist())
                    this_z_mu = z_mu[i] * r_C
                    this_z_sigma = z_sigma[i] * r_C
                    label_dist[p]["mu"].append(this_z_mu.mean().item())
                    label_dist[p]["sigma"].append(this_z_sigma.mean().item())

        classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        # print(label_dist)

        for label in label_dist:
            dist = label_dist[label]
            mus = np.array(dist["mu"])
            sigmas = np.array(dist["sigma"])
            x_, y_ = z[:, 0], z[:, 1]
            this_r_mu = r_mu[label].mean().item()
            this_r_sigma = r_sigma[label].mean().item()
            plt.clf()
            plt.scatter(x_, y_, label=classes[label], s=20)
            plt.axhline(y=this_r_sigma, linestyle="--", label="$r_\mu$")
            plt.axvline(x=this_r_mu, linestyle="--", label="$r_\sigma$")
            plt.xlabel("$\mu$")
            plt.ylabel("$\sigma$")
            cls = classes[label]
            plt.legend()
            plt.savefig(os.path.join(out_path, f"{cls}.png"))
