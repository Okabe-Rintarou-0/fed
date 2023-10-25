from argparse import Namespace
import copy
from typing import Any, Dict, List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
import torch.nn.functional as F
import numpy as np
from torch import nn

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_weights


class FedSRPlus3Server(FedServerBase):
    def __init__(
        self,
        args: Namespace,
        global_model: FedModel,
        clients: List[FedClientBase],
        writer: SummaryWriter | None = None,
    ):
        super().__init__(args, global_model, clients, writer)
        self.global_model.add_module("r", Rzy(args.num_classes, args.z_dim, args.m1))
        self.client_aggregatable_weights = global_model.get_aggregatable_weights()
        self.client_aggregatable_weights += ["r.C", "r.sigma", "r.mu", "r.pi"]

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedSR Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
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
        global_weight = aggregate_weights(
            local_weights, agg_weights, self.client_aggregatable_weights
        )
        # update global model
        self.global_model.load_state_dict(global_weight)

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


class Rzy(nn.Module):
    def __init__(self, num_classes: int, z_dim: int, n_components: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(num_classes, n_components, z_dim))
        self.sigma = nn.Parameter(torch.ones(num_classes, n_components, z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.pi = nn.Parameter(torch.ones(num_classes, n_components))


class FedSRPlus3Client(FedClientBase):
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
        assert (
            args.prob
        ), "FedSR only support probabilistic model, please use '--prob' flag to start it."
        self.l2r_coeff = args.l2r_coeff
        self.cmi_coeff = args.cmi_coeff
        self.r = Rzy(args.num_classes, args.z_dim, args.m1)
        self.can_agg_weights = self.local_model.get_aggregatable_weights() + [
            "r.C",
            "r.sigma",
            "r.mu",
            "r.pi",
        ]

        self.local_model.add_module("r", self.r)
        # Set optimizer for the local updates
        self.optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )
        self.local_model = self.local_model.to(self.device)
        self.n_components = args.m1

    def update_local_model(self, global_weight: Dict[str, Any]):
        local_weight = self.local_model.state_dict()
        for k in global_weight.keys():
            if k in self.can_agg_weights:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model
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
                    r_mus = self.r.mu[y]
                    r_sigmas = r_sigma_softplus[y]
                    r_pis = self.r.pi[y]
                    z_mu_scaled = z_mu * self.r.C
                    z_sigma_scaled = z_sigma * self.r.C
                    z_pi = 1 / self.n_components
                    reg_CMI = 0
                    item1 = 0
                    item2 = 0
                    for m1 in range(self.n_components):
                        r_mu = r_mus[:, m1]
                        r_sigma = r_sigmas[:, m1]
                        r_pi = r_pis[:, m1]
                        r_pi = F.softmax(r_pi, dim=0)
                        item1 = z_pi * torch.log(z_pi / r_pi)
                        item2 = z_pi * (
                            torch.log(r_sigma)
                            - torch.log(z_sigma_scaled)
                            + (z_sigma_scaled**2 + (z_mu_scaled - r_mu) ** 2)
                            / (2 * r_sigma**2)
                            - 0.5
                        ).sum(1)
                        this_reg_CMI = item1 + item2
                        reg_CMI += this_reg_CMI.mean()

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

        return result
