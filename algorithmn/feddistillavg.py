from argparse import Namespace
import copy
from typing import List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from algorithmn.base import FedClientBase, FedServerBase
from torch import nn
import numpy as np

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_weights


class FedDistillAvgServer(FedServerBase):
    def __init__(
        self,
        args: Namespace,
        global_model: FedModel,
        clients: List[FedClientBase],
        writer: SummaryWriter | None = None,
    ):
        super().__init__(args, global_model, clients, writer)
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()
        self.global_weight = self.global_model.state_dict()

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedDistillAvg Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        global_weight = self.global_weight
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
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_loss = result.loss_map["round_loss"]
            local_acc1 = result.acc_map["acc1"]
            local_acc2 = result.acc_map["acc2"]

            local_weights.append(copy.deepcopy(w))
            local_losses.append(local_loss)
            local_acc1s.append(local_acc1)
            local_acc2s.append(local_acc2)

            if not local_client.het_model:
                non_het_model_acc2s.append(local_acc2)

            acc1_dict[f"client_{idx}"] = local_acc1
            acc2_dict[f"client_{idx}"] = local_acc2
            loss_dict[f"client_{idx}"] = local_loss

        # get global weights
        self.global_weight = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights
        )

        loss_avg = sum(local_losses) / len(local_losses)
        non_het_model_acc2_avg = sum(non_het_model_acc2s) / len(non_het_model_acc2s)
        acc_avg1 = sum(local_acc1s) / len(local_acc1s)
        acc_avg2 = sum(local_acc2s) / len(local_acc2s)

        result = GlobalTrainResult(
            loss_map={
                "loss_avg": loss_avg,
            },
            acc_map={
                "non_het_model_acc2_avg": non_het_model_acc2_avg,
                "acc_avg1": acc_avg1,
                "acc_avg2": acc_avg2,
            },
        )
        if self.writer is not None:
            self.writer.add_scalars("clients_acc1", acc1_dict, round)
            self.writer.add_scalars("clients_acc2", acc2_dict, round)
            self.writer.add_scalars("clients_loss", loss_dict, round)
            self.writer.add_scalars("server_acc_avg", result.acc_map, round)
            self.writer.add_scalar("server_loss_avg", loss_avg, round)
        return result


class FedDistillAvgClient(FedClientBase):
    def __init__(
        self,
        idx: int,
        args: Namespace,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_model: FedModel,
        writer: SummaryWriter | None = None,
        het_model=False,
        teacher_model=None,
    ):
        super().__init__(
            idx,
            args,
            train_loader,
            test_loader,
            local_model,
            writer,
            het_model,
            teacher_model,
        )
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        acc1 = self.local_test()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )

        if self.teacher_model is not None:
            teacher_model = self.teacher_model
            # update teacher weights
            with torch.no_grad():
                teacher_weights = teacher_model.state_dict()
                local_weights = model.state_dict()
                for key in model.classifier_weight_keys:
                    teacher_weights[key] = local_weights[key]
                teacher_model.load_state_dict(teacher_weights)

            # train teacher first
            teacher_optimizer = torch.optim.SGD(
                teacher_model.parameters(),
                lr=self.args.lr,
                momentum=0.5,
                weight_decay=0.0005,
            )
            teacher_model.train()
            for _ in range(local_epoch):
                for images, labels in self.train_loader:
                    teacher_model.zero_grad()
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, output = teacher_model(images)
                    output = F.softmax(output, dim=1)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    teacher_optimizer.step()

        for _ in range(local_epoch):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)

                if self.teacher_model is not None:
                    teacher_model.eval()
                    _, teacher_output = model(images)
                    teacher_output = F.softmax(output, dim=1)

                    lam = self.args.distill_lambda
                    output = F.softmax(output, dim=1)
                    loss0 = self.criterion(output, labels)
                    loss1 = self.kl(output, teacher_output)
                    loss = lam * loss0 + (1 - lam) * loss1
                else:
                    loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc2 = self.local_test()

        result.weights = model.state_dict()
        result.acc_map["acc1"] = acc1
        result.acc_map["acc2"] = acc2
        round_loss = np.sum(round_losses) / len(round_losses)
        result.loss_map["round_loss"] = round_loss
        print(
            f"[client {self.idx}] local train acc: {result.acc_map}, loss: {result.loss_map}"
        )

        if self.writer is not None:
            self.writer.add_scalars(f"client_{self.idx}_acc", result.acc_map, round)
            self.writer.add_scalar(f"client_{self.idx}_loss", round_loss, round)

        return result