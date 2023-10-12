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
from tools import aggregate_weights, get_protos


class FedL2RegServer(FedServerBase):
    def __init__(self, args: Namespace, global_model: nn.Module, clients: List[FedClientBase], writer: SummaryWriter | None = None):
        super().__init__(args, global_model, clients, writer)

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f'\n---- Lg_FedAvg Global Communication Round : {round} ----')
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if (round >= self.args.epochs):
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        global_weight = self.global_model.state_dict()
        agg_weights = []
        local_weights = []
        local_losses = []
        local_acc1s = []
        local_acc2s = []

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        for idx in idx_clients:
            local_client: FedClientBase = self.clients[idx]
            agg_weights.append(local_client.agg_weight())
            local_epoch = self.args.local_epoch
            local_client.update_local_model(global_weight=global_weight)
            result = local_client.local_train(
                local_epoch=local_epoch, round=round)
            w = result.weights
            local_loss = result.loss_map['round_loss']
            local_acc1 = result.acc_map['acc1']
            local_acc2 = result.acc_map['acc2']

            local_weights.append(copy.deepcopy(w))
            local_losses.append(local_loss)
            local_acc1s.append(local_acc1)
            local_acc2s.append(local_acc2)

            acc1_dict[f'client_{idx}'] = local_acc1
            acc2_dict[f'client_{idx}'] = local_acc2
            loss_dict[f'client_{idx}'] = local_loss

        # get global weights
        global_weight = aggregate_weights(local_weights, agg_weights)
        # update global model
        self.global_model.load_state_dict(global_weight)

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


class FedL2RegClient(FedClientBase):
    def __init__(self, idx: int, args: Namespace, train_loader: DataLoader, test_loader: DataLoader, local_model: nn.Module, writer: SummaryWriter | None = None):
        super().__init__(idx, args, train_loader, test_loader, local_model, writer)
        self.w_local_keys = self.local_model.classifier_weight_keys

    def update_local_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def get_local_protos(self):
        model = self.local_model
        local_protos_list = {}
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, _ = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i,:])
                else:
                    local_protos_list[labels[i].item()] = [protos[i,:]]
        local_protos = get_protos(local_protos_list)
        return local_protos

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f'[client {self.idx}] local train round {round}:')
        model = self.local_model
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        global_protos = self.global_protos
        acc1 = self.local_test()

        self.last_model = copy.deepcopy(model)
        # get local prototypes before training, dict:={label: list of sample features}
        local_protos1 = self.get_local_protos()

        epoch_classifier = 1
        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        lr_g = 0.1
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_g,
                                    momentum=0.5, weight_decay=0.0005)
        for _ in range(epoch_classifier):
            # local training for 1 epoch
            data_loader = iter(self.train_data)
            iter_num = len(data_loader)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(
                    self.device), labels.to(self.device)
                model.zero_grad()
                protos, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            round_loss.append(sum(iter_loss)/len(iter_loss))
            iter_loss = []

        acc1, _ = self.local_test(self.test_data)

        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                    momentum=0.5, weight_decay=0.0005)

        local_epoch += epoch_classifier
        for _ in range(local_epoch):
            data_loader = iter(self.train_data)
            iter_num = len(data_loader)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                protos, output = model(images)
                loss0 = self.criterion(output, labels)
                loss1 = 0
                if round > 0:
                    loss1 = 0
                    protos_new = protos.clone().detach()
                    for i in range(len(labels)):
                        yi = labels[i].item()
                        if yi in global_protos:
                            protos_new[i] = global_protos[yi].detach()
                        else:
                            protos_new[i] = local_protos1[yi].detach()
                    loss1 = self.mse_loss(protos_new, protos)
                loss = loss0 + self.args.lam * loss1
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            round_loss.append(sum(iter_loss)/len(iter_loss))
            iter_loss = []

        acc2 = self.local_test()
        local_protos2 = self.get_local_protos()

        result.protos = local_protos2
        result.weights = model.state_dict()
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
