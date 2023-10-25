import argparse

import matplotlib.pyplot as plt
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
args = parser.parse_args()

if __name__ == "__main__":
    dataset = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device

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
        label_dist = {
            i: {"mu": [], "sigma": [], "kl_div": 0, "z": []}
            for i in range(7)
        }

        for env in range(len(dataset.ENVIRONMENTS)):
            this_dataset = dataset[env]
            dataloader = DataLoader(dataset=this_dataset, batch_size=batch_size)
            enough = False
            for images, labels in dataloader:
                if enough:
                    break
                images, labels = images.to(device), labels.to(device)
                z, output, (z_mu, z_sigma) = model(images, return_dist=True)
                predict = torch.argmax(output, dim=1)
                r_sigma = r_sigmas[labels]
                r_mu = r_mus[labels]
                for i in range(len(predict)):
                    p = predict[i].item()
                    label_dist[p]["z"].append(z[i].cpu().detach().tolist())
                    if len(label_dist[p]["z"]) > 300:
                        enough = True
                        break
                    # this_z_mu = z_mu[i] * r_C
                    # this_z_sigma = z_sigma[i] * r_C
                    # label_dist[p]["mu"].append(this_z_mu.mean().item())
                    # label_dist[p]["sigma"].append(this_z_sigma.mean().item())
                    # label_dist[p]["kl_div"] += (
                    #     (
                    #         torch.log(r_sigma[i])
                    #         - torch.log(this_z_sigma)
                    #         + (this_z_sigma**2 + (this_z_mu - r_mu[i]) ** 2)
                    #         / (2 * r_sigma[i] ** 2)
                    #         - 0.5
                    #     )
                    #     .sum()
                    #     .item()
                    # )

        classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        # print(label_dist)
        plt.xlabel("$\mu$")
        plt.ylabel("$\sigma$")

        for label in label_dist:
            dist = label_dist[label]
            # mus = np.array(dist["mu"])
            # sigmas = np.array(dist["sigma"])

            z = dist["z"]
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)

            # total = len(mus)
            # if total > 500:
            #     sampled = random.sample(list(range(total)), 500)
            # else:
            #     sampled = list(range(total))
            x_, y_ = z[:, 0], z[:, 1]
            plt.scatter(x_, y_, label=classes[label], s=20)
        plt.legend()
        plt.savefig("result.png")