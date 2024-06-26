from copy import deepcopy
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from algorithmn.base import FedClientBase
from algorithmn.fedavg import FedAvgClient, FedAvgServer
from algorithmn.fedclassavg import FedClassAvgClient, FedClassAvgServer
from algorithmn.fedl2reg import FedL2RegClient, FedL2RegServer
from algorithmn.fedprox import FedProxServer, FedProxClient
from algorithmn.fedts import FedTSClient, FedTSServer
from algorithmn.pfedgraph import PFedGraphClient, PFedGraphServer

from data_loader import (
    get_dataloaders,
    get_dataloaders_from_json,
    get_models,
)
from options import parse_args
from tensorboardX import SummaryWriter

from tools import (
    # draw_label_dist,
    write_client_datasets,
    write_client_label_distribution,
)

FL_CLIENT = {
    "FedAvg": FedAvgClient,
    "FedClassAvg": FedClassAvgClient,
    "FedProx": FedProxClient,
    "FedL2Reg": FedL2RegClient,
    "pFedGraph": PFedGraphClient,
    "FedTS": FedTSClient,
}

FL_SERVER = {
    "FedClassAvg": FedClassAvgServer,
    "FedAvg": FedAvgServer,
    "FedProx": FedProxServer,
    "FedL2Reg": FedL2RegServer,
    "pFedGraph": PFedGraphServer,
    "FedTS": FedTSServer,
}

LOADER_PATH_MAP = {
    "fmnist": {
        "train": "fmnist_train_client_20_dirichlet.json",
        "test": "fmnist_test_client_20_dirichlet.json",
    },
    "mnist": {
        "train": "mnist_train_client_20_dirichlet.json",
        "test": "mnist_test_client_20_dirichlet.json",
    },
    "cifar": {
        "train": "cifar_train_client_20_dirichlet.json",
        "test": "cifar_test_client_20_dirichlet.json",
    },
    "cinic10": {
        "train": "cinic10_train_client_20_dirichlet.json",
        "test": "cinic10_test_client_20_dirichlet.json",
    },
}

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    if dataset not in LOADER_PATH_MAP:
        train_loaders, test_loaders = get_dataloaders(args)
    else:
        train_path = os.path.join(
            f"./train_cfg/beta_{args.beta}", LOADER_PATH_MAP[dataset]["train"]
        )
        test_path = os.path.join(
            f"./train_cfg/beta_{args.beta}", LOADER_PATH_MAP[dataset]["test"]
        )
        train_loaders, test_loaders = get_dataloaders_from_json(
            args, train_path, test_path
        )

    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    student_model, teacher_model = get_models(args)

    train_rule = args.train_rule
    # Set up tensorboard summary writer
    sub_dir_name = train_rule
    if args.prob:
        sub_dir_name = f"{sub_dir_name}_prob"
    if not args.iid:
        sub_dir_name = f"{sub_dir_name}_non_iid"
    if args.model_het:
        sub_dir_name = f"{sub_dir_name}_model_het"

    sub_dir_name = f"{sub_dir_name}_{args.dataset}_te_{args.teacher_percent}_beta_{args.beta}_backbone_{args.backbone}"

    tensorboard_path = os.path.join(args.base_dir, "tensorboard", sub_dir_name)
    i = 1
    while os.path.exists(tensorboard_path):
        i += 1
        tensorboard_path = os.path.join(
            args.base_dir, "tensorboard", f"{sub_dir_name}_{i}"
        )

    writer = SummaryWriter(log_dir=tensorboard_path)

    # setup training data dir
    training_data_dir = os.path.join(
        args.base_dir, "training_data", f"{sub_dir_name}_{i}"
    )
    weights_dir = os.path.join(training_data_dir, "weights")
    training_data_json = os.path.join(training_data_dir, "data.json")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    training_args_json = os.path.join(training_data_dir, "args.json")
    with open(training_args_json, "w") as f:
        args_dict = vars(args)
        f.write(json.dumps(args_dict, indent=2))

    if train_rule not in FL_CLIENT or train_rule not in FL_SERVER:
        raise NotImplementedError()

    Client = FL_CLIENT[train_rule]
    Server = FL_SERVER[train_rule]

    # Training
    train_losses, train_accs = [], []
    test_accs = []
    local_accs1, local_accs2 = [], []
    local_clients = []

    client_idxs = list(range(args.stu_idx, args.num_clients))

    client_idx_set = set(client_idxs)
    teacher_num = int(args.teacher_percent * args.num_clients)
    teacher_clients = list(range(teacher_num))
    args.teacher_clients = teacher_clients
    print("teacher clients:", teacher_clients)

    args.num_clients = args.num_clients - args.stu_idx

    dists = []
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
                writer=writer,
            )

            if args.record_client_data:
                write_client_datasets(
                    idx, writer, train_loader, True, args.get_index)
                write_client_datasets(
                    idx, writer, test_loader, False, args.get_index)
                write_client_label_distribution(
                    idx, writer, train_loader, args.num_classes, args.get_index
                )
            local_clients.append(client)
            dists.append(client.label_distribution())
            bar.update(1)
    # draw_label_dist(dists, args.num_classes)

    server = Server(
        args=args, global_model=student_model, clients=local_clients, writer=writer
    )

    start_round = args.start_round
    end_round = args.epochs
    for round in range(start_round, end_round):
        train_result = server.train_one_round(round)
        round_loss = train_result.loss_map["loss_avg"]
        acc_avg = train_result.acc_map["acc_avg"]
        train_losses.append(round_loss)
        print(f"Train Loss: {round_loss}")
        print(
            f"Local Accuracy on Local Data: {acc_avg:.2f}%")

        # save client weights every 5 epochs
        if (round + 1) % 5 == 0:
            for idx in range(args.num_clients):
                weights_path = os.path.join(
                    weights_dir, f"client_{idx}_ckpt_{round}.pth"
                )
                local_client: FedClientBase = local_clients[idx]
                torch.save(local_client.local_model.state_dict(), weights_path)
            weights_path = os.path.join(
                weights_dir, f"global_ckpt_{round}.pth")
            torch.save(server.global_model.state_dict(), weights_path)
