import copy
from argparse import Namespace
from typing import List

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from algorithmn.base import FedClientBase, FedServerBase
from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_weights, weight_flatten


class FedProxServer(FedServerBase):
    def __init__(
        self,
        args: Namespace,
        global_model: FedModel,
        clients: List[FedClientBase],
        writer: SummaryWriter | None = None,
    ):
        super().__init__(args, global_model, clients, writer)
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()
        self.teacher_clients = args.teacher_clients

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedProx Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        agg_weights = []
        local_weights = []

        student_weights = []
        student_agg_weights = []

        teacher_weights = []
        teacher_agg_weights = []

        local_losses = []
        local_accs = []

        acc_dict = {}
        loss_dict = {}

        for idx in idx_clients:
            local_client: FedClientBase = self.clients[idx]
            local_epoch = self.args.local_epoch
            result = local_client.local_train(
                local_epoch=local_epoch, round=round)
            w = copy.deepcopy(result.weights)
            local_loss = result.loss_map["round_loss"]
            local_acc = result.acc_map["acc"]

            agg_weight = local_client.agg_weight()
            agg_weights.append(agg_weight)
            local_weights.append(w)
            if idx in self.teacher_clients:
                teacher_weights.append(w)
                teacher_agg_weights.append(agg_weight)
            else:
                student_weights.append(w)
                student_agg_weights.append(agg_weight)

            local_losses.append(local_loss)
            local_accs.append(local_acc)

            acc_dict[f"client_{idx}"] = local_acc
            loss_dict[f"client_{idx}"] = local_loss

        # get global weights
        classfier_weights = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights
        )

        student_weights = aggregate_weights(
            student_weights, student_agg_weights)
        teacher_weights = aggregate_weights(
            teacher_weights, teacher_agg_weights)

        for local_client in self.clients:
            if local_client.idx in self.teacher_clients:
                local_client.update_base_model(teacher_weights)
            else:
                local_client.update_base_model(student_weights)

            local_client.update_local_classifier(classfier_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg = sum(local_accs) / len(local_accs)

        result = GlobalTrainResult(
            loss_map={
                "loss_avg": loss_avg,
            },
            acc_map={
                "acc_avg": acc_avg,
            },
        )

        if self.args.model_het:
            self.analyze_hm_losses(
                idx_clients,
                local_losses,
                local_accs,
                result,
                self.teacher_clients,
            )

        if self.writer is not None:
            self.writer.add_scalars("clients_acc", acc_dict, round)
            self.writer.add_scalars("clients_loss", loss_dict, round)
            self.writer.add_scalars("server_acc_avg", result.acc_map, round)
            self.writer.add_scalars("server_loss_avg", result.loss_map, round)
        return result


class FedProxClient(FedClientBase):
    def __init__(
        self,
        idx: int,
        args: Namespace,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_model: FedModel,
        writer: SummaryWriter | None = None,
    ):
        super().__init__(
            idx,
            args,
            train_loader,
            test_loader,
            local_model,
            writer,
        )
        self.mu = args.mu
        self.mse_loss = nn.MSELoss()

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )

        with torch.no_grad():
            org_weights = (
                weight_flatten(model.state_dict()
                               ).detach().clone().to(self.device)
            )

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            for _ in range(len(data_loader)):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                loss = self.criterion(output, labels)
                cur_weights = weight_flatten(
                    model.state_dict()).to(self.device)
                loss += self.mu * self.mse_loss(cur_weights, org_weights)
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc = self.local_test()

        result.weights = model.state_dict()
        result.acc_map["acc"] = acc
        round_loss = np.sum(round_losses) / len(round_losses)
        result.loss_map["round_loss"] = round_loss
        print(
            f"[client {self.idx}] local train acc: {result.acc_map}, loss: {result.loss_map}"
        )

        return result
