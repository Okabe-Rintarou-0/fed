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

from datasets import PACS
from models.resnet import PACSResNet

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="pacs")
parser.add_argument("--weights", type=str)
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--out", default="./out")
args = parser.parse_args()


def confidence_ellipse(mu, cov, confidence=0.8, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a confidence ellipse for a given mean and covariance matrix.

    Parameters:
        mu (array-like): Mean of the distribution.
        cov (array-like): Covariance matrix of the distribution.
        confidence (float): Confidence level of the ellipse. Default is 0.8.
        n_std (float): The number of standard deviations to determine the size of the ellipse. Default is 3.0.
        facecolor (str): Color of the ellipse. Default is 'none'.
        **kwargs: Additional arguments to be passed to the Ellipse object.

    Returns:
        Ellipse: Ellipse object representing the confidence ellipse.
    """
    dof = mu.shape[0]  # Degrees of freedom (number of dimensions)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    chi2_val = chi2.ppf(confidence, df=dof)
    width, height = 2 * np.sqrt(chi2_val) * np.sqrt(eigenvalues)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])

    ellipse = plt.matplotlib.patches.Ellipse(
        xy=mu,
        width=width * n_std,
        height=height * n_std,
        angle=np.degrees(angle),
        facecolor=facecolor,
        **kwargs,
    )
    return ellipse


if __name__ == "__main__":
    dataset = args.dataset
    weights_path = args.weights
    batch_size = args.batch_size
    device = args.device
    out_path = args.out

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pca = PCA(n_components=2)
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
                for i in range(len(predict)):
                    p = predict[i].item()
                    label_dist[p]["z"].append(z[i].cpu().detach().tolist())
                    this_z_mu = z_mu[i] * r_C
                    this_z_sigma = z_sigma[i] * r_C
                    label_dist[p]["mu"].append(this_z_mu.cpu().detach().numpy())
                    label_dist[p]["sigma"].append(this_z_sigma.cpu().detach().numpy())

        classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        # print(label_dist)

        for label in label_dist:
            dist = label_dist[label]
            # mus = np.array(dist["mu"])
            # sigmas = np.array(dist["sigma"])
            zs = np.array(label_dist[p]["z"])
            zs = pca.fit_transform(zs)
            r_mu = r_mus[label].cpu().detach().numpy()
            r_sigma = r_sigmas[label].cpu().detach().numpy()
            samples = np.random.multivariate_normal(r_mu, np.diag(r_sigma), 1000)
            samples = pca.fit_transform(samples)
            r_mu = np.mean(samples, axis=0)
            r_sigma = np.cov(samples, rowvar=False)

            plt.clf()
            # Plot the 80% confidence interval ellipse
            ellipse = confidence_ellipse(
                r_mu,
                r_sigma,
                confidence=0.8,
                edgecolor="lightcoral",
                label="80% Confidence Interval",
            )
            plt.scatter(
                samples[:, 0],
                samples[:, 1],
                alpha=0.2,
                label="Samples of $r(z|y)$",
                s=10,
            )
            plt.scatter(zs[:, 0], zs[:, 1], alpha=0.5, label="Samples of $p(z|y)$")
            plt.gca().add_patch(ellipse)
            # plt.xlabel("$\mu$")
            # plt.ylabel("$\sigma$")
            cls = classes[label]
            plt.legend()
            plt.savefig(os.path.join(out_path, f"{cls}.png"))
