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
from tools import aggregate_protos, aggregate_weights, get_protos


class FedL2RegServer(FedServerBase):
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
        print(f"\n---- FedL2Reg Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        agg_weights = []
        local_weights = []
        local_losses = []
        local_accs = []
        local_protos = []
        label_sizes = []

        student_weights = []
        student_agg_weights = []

        teacher_weights = []
        teacher_agg_weights = []

        acc_dict = {}
        loss_dict = {}

        for idx in idx_clients:
            local_client: FedClientBase = self.clients[idx]
            local_epoch = self.args.local_epoch
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = copy.deepcopy(result.weights)
            local_loss = result.loss_map["round_loss"]
            local_acc = result.acc_map["acc"]

            local_losses.append(local_loss)
            local_accs.append(local_acc)
            local_protos.append(result.protos)
            label_sizes.append(local_client.label_cnts)

            agg_weight = local_client.agg_weight()
            agg_weights.append(agg_weight)
            local_weights.append(w)
            if idx in self.teacher_clients:
                teacher_weights.append(w)
                teacher_agg_weights.append(agg_weight)
            else:
                student_weights.append(w)
                student_agg_weights.append(agg_weight)

            acc_dict[f"client_{idx}"] = local_acc
            loss_dict[f"client_{idx}"] = local_loss

        # get global weights
        classifier_weights = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights
        )
        # update global prototype
        global_protos = aggregate_protos(local_protos, label_sizes)
        student_weights = aggregate_weights(student_weights, student_agg_weights)
        teacher_weights = aggregate_weights(teacher_weights, teacher_agg_weights)

        for local_client in self.clients:
            if local_client.idx in self.teacher_clients:
                local_client.update_base_model(teacher_weights)
            else:
                local_client.update_base_model(student_weights)
            local_client.update_local_classifier(classifier_weights)
            local_client.update_global_protos(global_protos=global_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg = sum(local_accs) / len(local_accs)

        result = GlobalTrainResult(
            loss_map={"loss_avg": loss_avg},
            acc_map={"acc_avg": acc_avg},
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


class FedL2RegClient(FedClientBase):
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
        self.mse_loss = nn.MSELoss()
        self.label_cnts = self.label_distribution()

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
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        global_protos = self.global_protos

        # get local prototypes before training, dict:={label: list of sample features}
        local_protos1 = self.get_local_protos()

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            iter_loss = []
            for _ in range(iter_num):
                images, labels = next(data_loader)
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
                loss2 = protos.norm(dim=1).mean()
                loss = loss0 + self.args.lam * loss1 + self.args.l2r_coeff * loss2
                loss.backward()
                max_grad_norm = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                iter_loss.append(loss.item())
            round_losses.append(sum(iter_loss) / len(iter_loss))

        acc = self.local_test()
        local_protos2 = self.get_local_protos()

        result.protos = local_protos2
        result.weights = model.state_dict()
        result.acc_map["acc"] = acc
        round_loss = np.sum(round_losses) / len(round_losses)
        result.loss_map["round_loss"] = round_loss
        print(
            f"[client {self.idx}] local train acc: {result.acc_map}, loss: {result.loss_map}"
        )

        # if self.writer is not None:
        #     self.writer.add_scalars(f"client_{self.idx}_acc", result.acc_map, round)
        #     self.writer.add_scalar(f"client_{self.idx}_loss", round_loss, round)

        return result
