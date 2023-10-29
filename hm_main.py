from copy import deepcopy
import json
import os
import random
from algorithmn.fedgen import FedGenClient, FedGenServer
from algorithmn.fedsrgen import FedSRGenClient, FedSRGenServer

from algorithmn.fedsrplus import FedSRPlusClient, FedSRPlusServer
import numpy as np
import torch
from tqdm import tqdm
from algorithmn.base import FedClientBase
from algorithmn.fedavg import FedAvgClient, FedAvgServer
from algorithmn.fedgmm import FedGMMClient, FedGMMServer
from algorithmn.fedl2reg import FedL2RegClient, FedL2RegServer
from algorithmn.fedpac import FedPACClient, FedPACServer
from algorithmn.fedper import FedPerClient, FedPerServer
from algorithmn.fedprox import FedProxServer, FedProxClient
from algorithmn.fedsr import FedSRClient, FedSRServer
from algorithmn.fedsrplus2 import FedSRPlus2Client, FedSRPlus2Server
from algorithmn.fedsrplus3 import FedSRPlus3Client, FedSRPlus3Server
from algorithmn.fedsrplus4 import FedSR4PlusServer, FedSRPlus4Client
from algorithmn.fedsrplus5 import FedSRPlus5Client, FedSRPlus5Server
from algorithmn.fedstandalone import FedStandAloneClient, FedStandAloneServer
from algorithmn.fedtts import FedTTSClient, FedTTSServer
from algorithmn.lg_fedavg import LgFedAvgClient, LgFedAvgServer
from algorithmn.pfedgraph import PFedGraphClient, PFedGraphServer

from data_loader import (
    get_dataloaders,
    get_model,
    get_models,
)
from options import parse_args
from tensorboardX import SummaryWriter

from tools import write_client_datasets, write_client_label_distribution

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
}


def write_training_data(training_data, training_data_json):
    with open(training_data_json, "w") as f:
        f.write(json.dumps(training_data))


def read_training_data(training_data_json):
    with open(training_data_json, "r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    args = parse_args()
    train_loaders, test_loaders = get_dataloaders(args)
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.model_het = True

    student_model, ta_model, teacher_model = get_models(args)

    train_rule = args.train_rule
    # Set up tensorboard summary writer
    sub_dir_name = train_rule
    if args.prob:
        sub_dir_name = f"{sub_dir_name}_prob"
    if not args.iid:
        sub_dir_name = f"{sub_dir_name}_non_iid"
    if args.domain_het:
        sub_dir_name = f"{sub_dir_name}_domain_het"
    if args.model_het:
        sub_dir_name = f"{sub_dir_name}_model_het"
    if args.attack:
        sub_dir_name = f"{sub_dir_name}_attack"

    sub_dir_name = f"{sub_dir_name}_{args.dataset}"
    tensorboard_path = os.path.join(args.base_dir, "tensorboard", sub_dir_name)
    writer = SummaryWriter(log_dir=tensorboard_path)

    # setup training data dir
    training_data_dir = os.path.join(args.base_dir, "training_data", sub_dir_name)
    weights_dir = os.path.join(training_data_dir, "weights")
    training_data_json = os.path.join(training_data_dir, "data.json")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if train_rule not in FL_CLIENT or train_rule not in FL_SERVER:
        raise NotImplementedError()

    Client = FL_CLIENT[train_rule]
    Server = FL_SERVER[train_rule]

    # Training
    train_losses, train_accs = [], []
    test_accs = []
    local_accs1, local_accs2 = [], []
    local_clients = []

    attack_clients = []
    client_idxs = list(range(args.num_clients))

    client_idx_set = set(client_idxs)
    ta_num = int(args.ta_percent * args.num_clients)
    ta_clients = random.sample(list(client_idx_set), ta_num)
    args.ta_clients = ta_clients
    print("ta clients:", ta_clients)

    client_idx_set -= set(ta_clients)
    teacher_num = int(args.teacher_percent * args.num_clients)
    teacher_clients = random.sample(list(client_idx_set), teacher_num)
    args.teacher_clients = teacher_clients
    print("teacher clients:", teacher_clients)

    if args.attack:
        sample_size = int(args.attack_percent * args.num_clients)
        attack_clients = random.sample(client_idxs, sample_size)
        args.attackers = attack_clients
        print(f"attack clients: {attack_clients}, attack type: {args.attack_type}")

    training_data = {
        "round": 0,
        "ta_clients": ta_clients,
        "teacher_clients": teacher_clients,
        "attack_clients": attack_clients,
        "attack_type": args.attack_type,
    }

    # write_training_data(
    #     training_data=training_data, training_data_json=training_data_json
    # )

    with tqdm(total=args.num_clients, desc="loading client") as bar:
        for idx in client_idxs:
            if idx in ta_clients:
                local_model = deepcopy(ta_model)
            elif idx in teacher_clients:
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
                write_client_datasets(idx, writer, train_loader, True, args.get_index)
                write_client_datasets(idx, writer, test_loader, False, args.get_index)
                write_client_label_distribution(
                    idx, writer, train_loader, args.num_classes, args.get_index
                )
            local_clients.append(client)
            bar.update(1)

    server = Server(
        args=args, global_model=student_model, clients=local_clients, writer=writer
    )

    start_round = args.start_round
    end_round = args.epochs
    for round in range(start_round, end_round):
        train_result = server.train_one_round(round)
        round_loss = train_result.loss_map["loss_avg"]
        local_acc1 = train_result.acc_map["acc_avg1"]
        local_acc2 = train_result.acc_map["acc_avg2"]
        train_losses.append(round_loss)
        print(f"Train Loss: {round_loss}")
        print(f"Local Accuracy on Local Data: {local_acc1:.2f}%, {local_acc2:.2f}%")
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)

        # save client weights every 5 epochs
        if (round + 1) % 5 == 0:
            for idx in range(args.num_clients):
                weights_path = os.path.join(
                    weights_dir, f"client_{idx}_ckpt_{round}.pth"
                )
                local_client: FedClientBase = local_clients[idx]
                torch.save(local_client.local_model.state_dict(), weights_path)
            training_data["round"] = round
            # write_training_data(
            #     training_data=training_data, training_data_json=training_data_json
            # )
            weights_path = os.path.join(weights_dir, f"global_ckpt_{round}.pth")
            torch.save(server.global_model.state_dict(), weights_path)
