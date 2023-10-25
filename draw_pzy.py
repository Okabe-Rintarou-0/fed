import argparse

import torch
from datasets import PACS

from models.resnet import PACSResNet
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="pacs")
parser.add_argument("--weights", type=str)
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch_size", default=32)
args = parser.parse_args()

if __name__ == "__main__":
    dataset = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device

    if dataset == "pacs":
        model = PACSResNet(num_classes=7, probabilistic=True).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        r_mu = state_dict["r.mu"]
        r_sigma = state_dict["r.sigma"]
        r_C = state_dict["r.C"]
        state_dict.pop("r.mu")
        state_dict.pop("r.sigma")
        state_dict.pop("r.C")

        model.load_state_dict(state_dict)
        dataset = PACS(root="./data", test_envs=[0])
        label_dist = {i: {"mu": [], "sigma": []} for i in range(7)}

        for env in range(len(dataset.ENVIRONMENTS)):
            this_dataset = dataset[env]
            dataloader = DataLoader(dataset=this_dataset, batch_size=batch_size)
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                _, output, (z_mu, z_sigma) = model(images, return_dist=True)
                predict = torch.argmax(output, dim=1)
                for i in range(len(predict)):
                    p = predict[i].item()
                    z_mu *= r_C
                    z_sigma *= r_C
                    label_dist[p]["mu"].extend(z_mu.mean(dim=1).tolist())
                    label_dist[p]["sigma"].extend(z_sigma.mean(dim=1).tolist())

        classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")
        for label in label_dist:
            dist = label_dist[label]
            mus = dist['mu']
            sigmas = dist['sigma']
            plt.scatter(mus, sigmas, label=classes[label])
        plt.legend()
        plt.savefig('result.png')
