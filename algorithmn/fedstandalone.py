from argparse import Namespace
import copy
from typing import List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
from torch import nn
import numpy as np

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel


class FedStandAloneServer(FedServerBase):
    def __init__(self, args: Namespace, global_model: FedModel, clients: List[FedClientBase], writer: SummaryWriter | None = None):
        super().__init__(args, global_model, clients, writer)

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(
            f'\n---- FedStandAlone Global Communication Round : {round} ----')
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if (round >= self.args.epochs):
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        local_losses = []
        local_acc1s = []
        local_acc2s = []

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        for idx in idx_clients:
            local_client: FedClientBase = self.clients[idx]
            local_epoch = self.args.local_epoch
            result = local_client.local_train(
                local_epoch=local_epoch, round=round)
            local_loss = result.loss_map['round_loss']
            local_acc1 = result.acc_map['acc1']
            local_acc2 = result.acc_map['acc2']

            local_losses.append(local_loss)
            local_acc1s.append(local_acc1)
            local_acc2s.append(local_acc2)

            acc1_dict[f'client_{idx}'] = local_acc1
            acc2_dict[f'client_{idx}'] = local_acc2
            loss_dict[f'client_{idx}'] = local_loss

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg1 = sum(local_acc1s) / len(local_acc1s)
        acc_avg2 = sum(local_acc2s) / len(local_acc2s)

        result = GlobalTrainResult(loss_map={
            'loss_avg': loss_avg
        }, acc_map={
            'acc_avg1': acc_avg1,
            'acc_avg2': acc_avg2
        })
        if self.writer is not None:
            self.writer.add_scalars('clients_acc1', acc1_dict, round)
            self.writer.add_scalars('clients_acc2', acc2_dict, round)
            self.writer.add_scalars('clients_loss', loss_dict, round)
            self.writer.add_scalars('server_acc_avg', result.acc_map, round)
            self.writer.add_scalar('server_loss_avg', loss_avg, round)
        return result


class FedStandAloneClient(FedClientBase):
    def __init__(self, idx: int, args: Namespace, train_loader: DataLoader, test_loader: DataLoader, local_model: FedModel, writer: SummaryWriter | None = None, het_model=False):
        super().__init__(idx, args, train_loader, test_loader, local_model, writer, het_model)

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f'[client {self.idx}] local train round {round}:')
        model = self.local_model
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        acc1 = self.local_test()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005)

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            for _ in range(len(data_loader)):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc2 = self.local_test()

        result.acc_map['acc1'] = acc1
        result.acc_map['acc2'] = acc2
        round_loss = np.sum(round_losses) / len(round_losses)
        result.loss_map['round_loss'] = round_loss
        print(
            f'[client {self.idx}] local train acc: {result.acc_map}, loss: {result.loss_map}')

        if self.writer is not None:
            self.writer.add_scalars(
                f"client_{self.idx}_acc", result.acc_map, round)
            self.writer.add_scalar(
                f"client_{self.idx}_loss", round_loss, round)

        return result
