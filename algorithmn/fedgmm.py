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
from tools import aggregate_weights


class FedGMMServer(FedServerBase):
    def __init__(self, args: Namespace, global_model: FedModel, clients: List[FedClientBase], writer: SummaryWriter | None = None):
        super().__init__(args, global_model, clients, writer)
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f'\n---- FedAvg Global Communication Round : {round} ----')
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if (round >= self.args.epochs):
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        global_weight = self.global_model.state_dict()
        non_het_model_acc2s = []
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

            if not local_client.het_model:
                non_het_model_acc2s.append(local_acc2)

            acc1_dict[f'client_{idx}'] = local_acc1
            acc2_dict[f'client_{idx}'] = local_acc2
            loss_dict[f'client_{idx}'] = local_loss

        # get global weights
        global_weight = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights)
        # update global model
        self.global_model.load_state_dict(global_weight)

        loss_avg = sum(local_losses) / len(local_losses)
        non_het_model_acc2_avg = sum(
            non_het_model_acc2s) / len(non_het_model_acc2s)
        acc_avg1 = sum(local_acc1s) / len(local_acc1s)
        acc_avg2 = sum(local_acc2s) / len(local_acc2s)

        result = GlobalTrainResult(loss_map={
            'loss_avg': loss_avg,
        }, acc_map={
            'non_het_model_acc2_avg': non_het_model_acc2_avg,
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


class FedGMMClient(FedClientBase):
    def __init__(self, idx: int, args: Namespace, train_loader: DataLoader, test_loader: DataLoader, local_model: FedModel, writer: SummaryWriter | None = None, het_model=False):
        super().__init__(idx, args, train_loader, test_loader, local_model, writer, het_model)
        self.M1 = args.m1
        z_dim = args.z_dim
        self.mu = np.zeros((self.M1, z_dim))
        self.sigma = np.zeros((self.M1, z_dim))
        self.pi = np.zeros((self.M1, z_dim))

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            torch.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.size()) == 3:
            x = x.squeeze(1)
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x[np.random.choice(
                np.arange(x.shape[0]), size=n_centers, replace=True), ...]
            l2_dis = torch.norm((x.unsqueeze(1).repeat(
                1, n_centers, 1) - tmp_center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)

            cost = 0
            for c in range(n_centers):
                if not (l2_cls == c).any():
                    continue
                cost += torch.norm(x[l2_cls == c] -
                                   tmp_center[c], p=2, dim=1).mean()

                assert not torch.isnan(cost)

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf

        while delta > min_delta:
            l2_dis = torch.norm((x.unsqueeze(1).repeat(
                1, n_centers, 1) - center), p=2, dim=2)
            l2_cls = torch.argmin(l2_dis, dim=1)
            center_old = center.clone()

            for c in range(n_centers):
                if not (l2_cls == c).any():
                    center[c] = center_old[c]
                    continue
                center[c] = x[l2_cls == c].mean(dim=0)

            delta = torch.norm((center_old - center), dim=1).max()

        return (center.unsqueeze(0) * (x_max - x_min) + x_min)

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

                # ---------------- Client E-stage ----------------- #
                for m1 in range(self.M1):
                    pass

                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc2 = self.local_test()

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
