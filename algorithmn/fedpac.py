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
from tools import agg_classifier_weighted_p, aggregate_protos, aggregate_weights, get_head_agg_weight, get_protos


class FedPACServer(FedServerBase):
    def __init__(self, args: Namespace, global_model: FedModel, clients: List[FedClientBase], writer: SummaryWriter | None = None):
        super().__init__(args, global_model, clients, writer)
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f'\n---- FedPAC Global Communication Round : {round} ----')
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if (round >= self.args.epochs):
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        agg_weights = []
        local_weights = []
        local_losses = []
        local_acc1s = []
        local_acc2s = []
        local_protos = []
        label_sizes = []

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        Vars = []
        Hs = []
        sizes_label = []

        for idx in idx_clients:
            local_client: FedPACClient = self.clients[idx]
            # statistics collection
            v, h = local_client.statistics_extraction()
            Vars.append(copy.deepcopy(v))
            Hs.append(copy.deepcopy(h))

            agg_weights.append(local_client.agg_weight())

            sizes_label.append(local_client.sizes_label)

            local_epoch = self.args.local_epoch
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
            local_protos.append(result.protos)
            label_sizes.append(local_client.label_distribution())

            acc1_dict[f'client_{idx}'] = local_acc1
            acc2_dict[f'client_{idx}'] = local_acc2
            loss_dict[f'client_{idx}'] = local_loss

        # get global weights
        global_weight = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights)
        # update global prototype
        global_protos = aggregate_protos(local_protos, label_sizes)

        for local_client in self.clients:
            local_client.update_base_model(global_weight=global_weight)
            local_client.update_global_protos(global_protos=global_protos)

        avg_weights = get_head_agg_weight(m, Vars, Hs)
        idxx = 0
        for idx in idx_clients:
            local_client = self.clients[idx]
            if avg_weights[idxx] is not None:
                new_cls = agg_classifier_weighted_p(
                    local_weights, avg_weights[idxx], local_client.w_local_keys, idxx)
            else:
                new_cls = local_weights[idxx]
            local_client.update_local_classifier(new_weight=new_cls)
            idxx += 1

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


class FedPACClient(FedClientBase):
    def __init__(self, idx: int, args: Namespace, train_loader: DataLoader, test_loader: DataLoader, local_model: FedModel, writer: SummaryWriter | None = None, het_model=False):
        super().__init__(idx, args, train_loader, test_loader, local_model, writer, het_model)
        self.mse_loss = nn.MSELoss()
        self.num_classes = args.num_classes
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.sizes_label = torch.tensor(
            self.label_distribution()).to(self.device)

        self.probs_label = (self.sizes_label /
                            len(self.train_loader.dataset)).to(self.device)
        self.datasize = torch.tensor(
            len(self.train_loader.dataset)).to(self.device)

    def statistics_extraction(self):
        model = self.local_model
        cls_keys = self.w_local_keys
        g_params = model.state_dict()[cls_keys[0]] if isinstance(
            cls_keys, list) else model.state_dict()[cls_keys]
        d = g_params[0].shape[0]
        feature_dict = {}
        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, _ = model(inputs)
                feat_batch = features.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = self.probs_label
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k]*feat_k_mu
                v += (py[k]*torch.trace((torch.mm(torch.t(feat_k), feat_k)/num_k))).item()
                v -= (py2[k]*(torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v/self.datasize.item()

        return v, h_ref

    def update_base_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def update_local_classifier(self, new_weight):
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k in w_local_keys:
                local_weight[k] = new_weight[k]
        self.local_model.load_state_dict(local_weight)

    def get_local_protos(self):
        model = self.local_model
        local_protos_list = {}
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, _ = model(inputs)
            protos = features.clone().detach()
            for i in range(len(labels)):
                if labels[i].item() in local_protos_list.keys():
                    local_protos_list[labels[i].item()].append(protos[i, :])
                else:
                    local_protos_list[labels[i].item()] = [protos[i, :]]
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

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                    momentum=0.5, weight_decay=0.0005)
        for _ in range(epoch_classifier):
            # local training for 1 epoch
            iter_loss = []
            for images, labels in self.train_loader:
                images, labels = images.to(
                    self.device), labels.to(self.device)
                model.zero_grad()
                protos, output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            round_losses.append(sum(iter_loss)/len(iter_loss))

        local_epoch += 1
        for name, param in model.named_parameters():
            if name in self.w_local_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                    momentum=0.5, weight_decay=0.0005)
        for _ in range(local_epoch):
            iter_loss = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                protos, output = model(images)
                loss0 = self.criterion(output, labels)
                loss1 = 0
                if round > 0:
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
            round_losses.append(sum(iter_loss)/len(iter_loss))

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
