from copy import deepcopy
import json
import os
import random
import numpy as np
from sklearn.decomposition import KernelPCA
import torch
from tqdm import tqdm
from algorithmn.fedgen import FedGenClient, FedGenServer
from algorithmn.base import FedClientBase
from algorithmn.fedavg import FedAvgClient, FedAvgServer
from algorithmn.fedgmm import FedGMMClient, FedGMMServer
from algorithmn.fedl2reg import FedL2RegClient, FedL2RegServer
from algorithmn.fedpac import FedPACClient, FedPACServer
from algorithmn.fedper import FedPerClient, FedPerServer
from algorithmn.fedprox import FedProxServer, FedProxClient
from algorithmn.fedsr import FedSRClient, FedSRServer
from algorithmn.fedstandalone import FedStandAloneClient, FedStandAloneServer
from algorithmn.fedtsgen import FedTSGenClient, FedTSGenServer
from algorithmn.fedtts import FedTTSClient, FedTTSServer
from algorithmn.lg_fedavg import LgFedAvgClient, LgFedAvgServer
from algorithmn.pfedgraph import PFedGraphClient, PFedGraphServer

import matplotlib.pyplot as plt

from data_loader import (
    get_dataloaders,
    get_dataloaders_from_json,
    get_models,
)
from options import parse_args
from tensorboardX import SummaryWriter

from tools import (
    draw_label_dist,
    write_client_datasets,
    write_client_label_distribution,
)

FL_CLIENT = {
    "FedStandAlone": FedStandAloneClient,
    "FedAvg": FedAvgClient,
    "FedProx": FedProxClient,
    "Lg_FedAvg": LgFedAvgClient,
    "FedPer": FedPerClient,
    "FedL2Reg": FedL2RegClient,
    "pFedGraph": PFedGraphClient,
    "FedSR": FedSRClient,
    "FedPAC": FedPACClient,
    "FedGMM": FedGMMClient,
    "FedTTS": FedTTSClient,
    "FedGen": FedGenClient,
    "FedTSGen": FedTSGenClient,
}

FL_SERVER = {
    "FedStandAlone": FedStandAloneServer,
    "FedAvg": FedAvgServer,
    "FedProx": FedProxServer,
    "Lg_FedAvg": LgFedAvgServer,
    "FedPer": FedPerServer,
    "FedL2Reg": FedL2RegServer,
    "pFedGraph": PFedGraphServer,
    "FedSR": FedSRServer,
    "FedPAC": FedPACServer,
    "FedGMM": FedGMMServer,
    "FedTTS": FedTTSServer,
    "FedGen": FedGenServer,
    "FedTSGen": FedTSGenServer,
}


def write_training_data(training_data, training_data_json):
    with open(training_data_json, "w") as f:
        f.write(json.dumps(training_data))


def read_training_data(training_data_json):
    with open(training_data_json, "r") as f:
        return json.loads(f.read())


LOADER_PATH_MAP = {
    "mnist": {
        "train": "./train_cfg/mnist_train_client_20_dirichlet.json",
        "test": "./train_cfg/mnist_test_client_20_dirichlet.json",
    },
    "emnist": {
        "train": "./train_cfg/emnist_train_client_20_dirichlet.json",
        "test": "./train_cfg/emnist_test_client_20_dirichlet.json",
    },
    "fmnist": {
        "train": "./train_cfg/fmnist_train_client_20_dirichlet.json",
        "test": "./train_cfg/fmnist_test_client_20_dirichlet.json",
    },
    "cifar": {
        "train": "./train_cfg/cifar_train_client_20_dirichlet.json",
        "test": "./train_cfg/cifar_test_client_20_dirichlet.json",
    },
    "cinic10": {
        "train": "./train_cfg/cinic10_train_client_20_dirichlet.json",
        "test": "./train_cfg/cinic10_test_client_20_dirichlet.json",
    },
    "cifar100": {
        "train": "./train_cfg/cifar100_train_client_20_dirichlet.json",
        "test": "./train_cfg/cifar100_test_client_20_dirichlet.json",
    },
}

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    if dataset not in LOADER_PATH_MAP:
        train_loaders, test_loaders = get_dataloaders(args)
    else:
        train_loaders, test_loaders = get_dataloaders_from_json(
            args, LOADER_PATH_MAP[dataset]["train"], LOADER_PATH_MAP[dataset]["test"]
        )
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.model_het = True

    student_model, ta_model, teacher_model = get_models(args)

    root = "/Volumes/upan/fedtsgen"
    client_weight_paths = [f"client_{i}_ckpt_199.pth" for i in range(20)]

    client_idxs = list(range(20))
    teacher_clients = list(range(8))

    train_rule = args.train_rule
    if train_rule not in FL_CLIENT or train_rule not in FL_SERVER:
        raise NotImplementedError()

    Client = FL_CLIENT[train_rule]
    local_clients = []
    with tqdm(total=args.num_clients, desc="loading client") as bar:
        for idx in client_idxs:
            if idx in teacher_clients:
                local_model = deepcopy(teacher_model)
            else:
                local_model = deepcopy(student_model)

            train_loader = train_loaders[idx]
            test_loader = test_loaders[idx]
            client = Client(
                idx=idx,
                args=args,
                train_loader=train_loader,
                test_loader=test_loader,
                local_model=local_model.to(args.device),
                writer=None,
            )

            client.local_model.load_state_dict(
                torch.load(
                    os.path.join(root, client_weight_paths[idx]), map_location="cpu"
                )
            )

            local_clients.append(client)
            bar.update(1)

    sum = 0
    latent_presents = []
    for client in local_clients:
        is_teacher = client.idx in teacher_clients
        print(f"client {client.idx}...", end="")
        for imgs, labels in client.train_loader:
            zs, _ = client.local_model(imgs)
            for idx, label in enumerate(labels):
                if label == 0:
                    latent_presents.append(zs[idx].detach().numpy())
                    if is_teacher:
                        sum += 1
        print("done")

    latent_presents = np.array(latent_presents)
    kpca = KernelPCA(n_components=2)
    latent_presents = kpca.fit_transform(latent_presents)
    x = latent_presents[:, 0]
    y = latent_presents[:, 1]

    plt.clf()
    plt.scatter(x[:sum], y[:sum], label="teacher", alpha=0.2, s=10)
    plt.scatter(x[sum:], y[sum:], label="student", alpha=0.2, s=10)
    plt.legend()
    plt.savefig("./st.png")
