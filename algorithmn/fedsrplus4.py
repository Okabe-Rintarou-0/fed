from argparse import Namespace
import copy
from typing import List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import distributions

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_weights


class FedSR4PlusServer(FedServerBase):
    def __init__(
        self,
        args: Namespace,
        global_model: FedModel,
        clients: List[FedClientBase],
        writer: SummaryWriter | None = None,
    ):
        super().__init__(args, global_model, clients, writer)
        self.global_model.add_module("r", Rzy(args.num_classes, args.z_dim))
        self.global_model.all_keys += ["r.C", "r.sigma", "r.mu"]
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedSR+ Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
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
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_loss = result.loss_map["round_loss"]
            local_acc1 = result.acc_map["acc1"]
            local_acc2 = result.acc_map["acc2"]

            local_weights.append(copy.deepcopy(w))
            local_losses.append(local_loss)
            local_acc1s.append(local_acc1)
            local_acc2s.append(local_acc2)

            acc1_dict[f"client_{idx}"] = local_acc1
            acc2_dict[f"client_{idx}"] = local_acc2
            loss_dict[f"client_{idx}"] = local_loss

        # get global weights
        global_weight = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights
        )
        # update global model
        self.global_model.load_state_dict(global_weight)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg1 = sum(local_acc1s) / len(local_acc1s)
        acc_avg2 = sum(local_acc2s) / len(local_acc2s)

        result = GlobalTrainResult(
            loss_map={
                "loss_avg": loss_avg,
            },
            acc_map={
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


class Rzy(nn.Module):
    def __init__(self, num_classes: int, z_dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(num_classes, z_dim))
        self.sigma = nn.Parameter(torch.ones(num_classes, z_dim))
        self.C = nn.Parameter(torch.ones([]))


class FedSRPlus4Client(FedClientBase):
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
        assert (
            args.prob
        ), "FedSR+ only support probabilistic model, please use '--prob' flag to start it."
        self.l2r_coeff = args.l2r_coeff
        self.cmi_coeff = args.cmi_coeff
        self.r = Rzy(args.num_classes, args.z_dim)
        self.local_model.add_module("r", self.r)
        self.available_labels = list(range(args.num_classes))
        self.gen_batch_size = args.gen_batch_size
        self.z_dim = args.z_dim
        label_cnt = torch.tensor(self.label_distribution(), device=self.device)
        self.label_weight = F.softmax(1 - (label_cnt / torch.sum(label_cnt)), dim=0)
        self.ce = nn.CrossEntropyLoss(reduction="none")

        # Set optimizer for the local updates
        self.optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )

        self.local_model.all_keys += ["r.C", "r.sigma", "r.mu"]
        self.local_model = self.local_model.to(self.device)

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model.to(self.device)
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        acc1 = self.local_test()

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            if self.args.iter_num > 0:
                iter_num = min(iter_num, self.args.iter_num)
            for _ in range(iter_num):
                y_sampled = torch.tensor(
                    np.random.choice(self.available_labels, self.gen_batch_size),
                    device=self.device,
                )

                label_weight = self.label_weight[y_sampled].view([1, -1])

                r_sigma_softplus = F.softplus(self.r.sigma)
                r_mu = self.r.mu[y_sampled].clone().detach()
                r_sigma = r_sigma_softplus[y_sampled].clone().detach()
                r_dist = distributions.Independent(
                    distributions.normal.Normal(r_mu, r_sigma), 1
                )
                r = r_dist.rsample([1]).view([-1, self.z_dim])
                z = r.to(self.device)
                y_predicted = self.local_model.classifier(z)
                this_loss = self.ce(y_predicted, y_sampled).unsqueeze(-1)
                loss = (label_weight @ this_loss).squeeze() / self.gen_batch_size
                loss.backward()
                self.optimizer.step()

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            if self.args.iter_num > 0:
                iter_num = min(iter_num, self.args.iter_num)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()

                z, logits, (z_mu, z_sigma) = model(images, return_dist=True)
                y = labels
                loss = self.criterion(logits, labels)

                if self.l2r_coeff != 0.0:
                    reg_L2R = z.norm(dim=1).mean()
                    loss += self.l2r_coeff * reg_L2R

                if self.cmi_coeff != 0.0:
                    r_sigma_softplus = F.softplus(self.r.sigma)
                    r_mu = self.r.mu[y]
                    r_sigma = r_sigma_softplus[y]
                    z_mu_scaled = z_mu * self.r.C
                    z_sigma_scaled = z_sigma * self.r.C
                    reg_CMI = (
                        torch.log(r_sigma)
                        - torch.log(z_sigma_scaled)
                        + (z_sigma_scaled**2 + (z_mu_scaled - r_mu) ** 2)
                        / (2 * r_sigma**2)
                        - 0.5
                    )
                    reg_CMI = reg_CMI.sum(1).mean()
                    loss += self.cmi_coeff * reg_CMI

                loss.backward()
                self.optimizer.step()
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

        self.clear_memory()
        return result
