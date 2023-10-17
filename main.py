from copy import deepcopy
import json
import os
import random
import numpy as np
import torch
from algorithmn.base import FedClientBase
from algorithmn.fedavg import FedAvgClient, FedAvgServer
from algorithmn.fedl2reg import FedL2RegClient, FedL2RegServer
from algorithmn.fedpac import FedPACClient, FedPACServer
from algorithmn.fedper import FedPerClient, FedPerServer
from algorithmn.fedsr import FedSRClient, FedSRServer
from algorithmn.fedstandalone import FedStandAloneClient, FedStandAloneServer
from algorithmn.lg_fedavg import LgFedAvgClient, LgFedAvgServer
from algorithmn.pfedgraph import PFedGraphClient, PFedGraphServer
from data_loader import get_dataloaders, get_heterogeneous_model, get_model
from options import parse_args
from tensorboardX import SummaryWriter

from tools import write_client_datasets, write_client_label_distribution

FL_CLIENT = {
    'FedStandAlone': FedStandAloneClient,
    'FedAvg': FedAvgClient,
    'Lg_FedAvg': LgFedAvgClient,
    'FedPer': FedPerClient,
    'FedL2Reg': FedL2RegClient,
    'pFedGraph': PFedGraphClient,
    'FedSR': FedSRClient,
    'FedPAC': FedPACClient
}

FL_SERVER = {
    'FedStandAlone': FedStandAloneServer,
    'FedAvg': FedAvgServer,
    'Lg_FedAvg': LgFedAvgServer,
    'FedPer': FedPerServer,
    'FedL2Reg': FedL2RegServer,
    'pFedGraph': PFedGraphServer,
    'FedSR': FedSRServer,
    'FedPAC': FedPACServer
}

if __name__ == '__main__':
    args = parse_args()
    train_loaders, test_loaders = get_dataloaders(args)
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    heterogeneous_model = None
    if args.model_het:
        heterogeneous_model = get_heterogeneous_model(args)

    # construct model
    global_model = get_model(args)

    train_rule = args.train_rule
    # Set up tensorboard summary writer
    sub_dir_name = train_rule
    if args.prob:
        sub_dir_name = f'{sub_dir_name}_prob'
    if not args.iid:
        sub_dir_name = f'{sub_dir_name}_non_iid'
    if args.model_het:
        sub_dir_name = f'{sub_dir_name}_model_het'

    tensorboard_path = os.path.join('./tensorboard', sub_dir_name)
    writer = SummaryWriter(log_dir=tensorboard_path)

    # setup training data dir
    training_data_dir = os.path.join('./training_data', sub_dir_name)
    weights_dir = os.path.join(training_data_dir, 'weights')
    training_data_json = os.path.join(training_data_dir, 'data.json')
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

    heterogeneous_clients = []
    client_idxs = list(range(args.num_clients))
    if args.model_het:
        sample_size = int(args.model_het_percent * args.num_clients)
        heterogeneous_clients = random.sample(client_idxs, sample_size)
        print('heterogeneous clients:', heterogeneous_clients)

    training_data = {
        'heterogeneous_clients': heterogeneous_clients
    }
    with open(training_data_json, 'w') as f:
        f.write(json.dumps(training_data))

    for idx in client_idxs:
        is_heterogeneous_client = idx in heterogeneous_clients
        if is_heterogeneous_client:
            local_model = deepcopy(heterogeneous_model)
        else:
            local_model = deepcopy(global_model)

        train_loader = train_loaders[idx]
        test_loader = test_loaders[idx]
        client = Client(idx=idx, args=args,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        local_model=local_model,
                        writer=writer,
                        het_model=is_heterogeneous_client
                        )
        write_client_datasets(idx, writer, train_loader, True)
        write_client_datasets(idx, writer, test_loader, False)
        write_client_label_distribution(
            idx, writer, train_loader, args.num_classes)
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

        # save client weights every 5 epochs
        if round % 5 == 0:
            for idx in range(args.num_clients):
                weights_path = os.path.join(weights_dir, f'client_{idx}.pth')
                local_client: FedClientBase = local_clients[idx]
                torch.save(local_client.local_model.state_dict(), weights_path)
