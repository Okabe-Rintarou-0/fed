from copy import deepcopy
import os
import numpy as np
import torch
from algorithmn.fedavg import FedAvgClient, FedAvgServer
from algorithmn.fedper import FedPerClient, FedPerServer
from algorithmn.fedstandalone import FedStandAloneClient, FedStandAloneServer
from algorithmn.lg_fedavg import LgFedAvgClient, LgFedAvgServer
from data_loader import get_dataloaders, get_model
from options import parse_args
from tensorboardX import SummaryWriter

from tools import write_client_datasets

FL_CLIENT = {
    'FedStandAlone': FedStandAloneClient,
    'FedAvg': FedAvgClient,
    'Lg_FedAvg': LgFedAvgClient,
    'FedPer': FedPerClient
}

FL_SERVER = {
    'FedStandAlone': FedStandAloneServer,
    'FedAvg': FedAvgServer,
    'Lg_FedAvg': LgFedAvgServer,
    'FedPer': FedPerServer
}

if __name__ == '__main__':
    args = parse_args()
    train_loaders, test_loaders = get_dataloaders(args)
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # construct model
    global_model = get_model(args)

    train_rule = args.train_rule
    # Set up tensorboard summary writer
    tensorboard_path = os.path.join('./tensorboard', train_rule)
    writer = SummaryWriter(log_dir=tensorboard_path)

    if train_rule not in FL_CLIENT or train_rule not in FL_SERVER:
        raise NotImplementedError()

    Client = FL_CLIENT[train_rule]
    Server = FL_SERVER[train_rule]

    # Training
    train_losses, train_accs = [], []
    test_accs = []
    local_accs1, local_accs2 = [], []
    local_clients = []
    for idx in range(args.num_clients):
        client = Client(idx=idx, args=args,
                        train_loader=train_loaders[idx],
                        test_loader=test_loaders[idx],
                        local_model=deepcopy(global_model),
                        writer=writer)
        write_client_datasets(idx, writer, train_loaders[idx], True)
        write_client_datasets(idx, writer, test_loaders[idx], False)
        local_clients.append(client)

    server = Server(args=args, global_model=global_model,
                    clients=local_clients, writer=writer)

    for round in range(args.epochs):
        train_result = server.train_one_round(round)
        round_loss = train_result.loss_map['loss_avg']
        local_acc1 = train_result.acc_map['acc_avg1']
        local_acc2 = train_result.acc_map['acc_avg2']
        train_losses.append(round_loss)
        print(f"Train Loss: {round_loss}")
        print(
            f"Local Accuracy on Local Data: {local_acc1:.2f}%, {local_acc2:.2f}%")
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)
