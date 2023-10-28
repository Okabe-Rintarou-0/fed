from argparse import Namespace
import copy
import random
from typing import Any, Dict, List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
import torch.nn.functional as F
import numpy as np
from torch import nn
from scipy.stats import norm

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_dist, aggregate_weights, get_protos


class FedSRPlus5Server(FedServerBase):
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

        local_mus = []
        local_sigmas = []
        local_avg_confs = []

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        for idx in idx_clients:
            local_client: FedSRPlus5Client = self.clients[idx]
            agg_weights.append(local_client.agg_weight())
            local_epoch = self.args.local_epoch
            local_client.update_local_model(global_weight=global_weight)
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_loss = result.loss_map["round_loss"]
            local_acc1 = result.acc_map["acc1"]
            local_acc2 = result.acc_map["acc2"]

            local_mus.append(local_client.local_mus.detach().clone())
            local_sigmas.append(local_client.local_sigmas.detach().clone())
            local_avg_confs.append(local_client.avg_conf.detach().clone())

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

        agg_mu, agg_sigma = aggregate_dist(
            local_mus, local_sigmas, local_avg_confs, agg_weights, self.args.num_classes
        )

        for client in self.clients:
            client.global_mus = agg_mu
            client.global_sigmas = agg_sigma

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
        self.sigma = nn.Parameter(torch.rand(num_classes, n_components, z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.pi = nn.Parameter(torch.rand(num_classes, n_components))


class FedSRPlus5Client(FedClientBase):
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
        self.local_mus = torch.zeros(
            (args.num_classes, args.z_dim), requires_grad=False
        )
        self.local_sigmas = torch.zeros(
            (args.num_classes, args.z_dim), requires_grad=False
        )
        self.avg_conf = torch.zeros((args.num_classes,), requires_grad=False)

        self.global_mus = torch.zeros(
            (args.num_classes, args.z_dim), requires_grad=False
        )
        self.global_sigmas = torch.zeros(
            (args.num_classes, args.z_dim), requires_grad=False
        )

        self.local_model.add_module("r", self.r)
        # Set optimizer for the local updates
        self.optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )
        self.local_model = self.local_model.to(self.device)
        self.n_components = args.m1

    def update_local_dist(self):
        model = self.local_model
        self.local_mus.zero_()
        self.local_sigmas.zero_()
        self.avg_conf.zero_()
        label_cnt = [0] * self.args.num_classes
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _, logits, (z_mu, z_sigma) = model(inputs, return_dist=True)

            z_mu = z_mu.cpu().detach()
            z_sigma = z_sigma.cpu().detach()

            for i in range(len(labels)):
                label = labels[i]
                label_cnt[label] += 1
                self.avg_conf[label] += logits[i][label]
                self.local_mus[label] += z_mu[i]
                self.local_sigmas[label] += z_sigma[i]

        for label in range(self.args.num_classes):
            cnt = label_cnt[label]
            if cnt == 0:
                continue
            self.local_mus[label] /= cnt
            self.local_sigmas[label] /= cnt
            self.avg_conf[label] /= cnt

    def update_local_model(self, global_weight: Dict[str, Any]):
        local_weight = self.local_model.state_dict()
        for k in global_weight.keys():
            if k in self.can_agg_weights:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model.to(self.device)
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        acc1 = self.local_test()

        epoch_classifier = int(round > 0)
        for name, param in model.named_parameters():
            if name in self.local_model.classifier_weight_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for _ in range(epoch_classifier):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            if self.args.iter_num > 0:
                iter_num = min(iter_num, self.args.iter_num)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                z, logits, (z_mu, z_sigma) = model(images, return_dist=True)

                batch_size = z.shape[0]

                rand_labels = torch.tensor(
                    np.random.choice(list(range(self.args.num_classes)), batch_size),
                    device=self.device,
                    dtype=torch.long
                )

                sampled_mus = self.global_mus[rand_labels]
                sampled_sigmas = self.global_sigmas[rand_labels]
                z2 = sampled_mus + sampled_sigmas * torch.randn(
                    batch_size, self.args.z_dim
                )

                lam = torch.rand((batch_size, 1))
                sampled_zs = lam * z.detach().clone() + (1 - lam) * z2
                d1 = (
                    (sampled_zs - z_mu).unsqueeze(1)
                    @ torch.diag_embed(1 / z_sigma)
                    @ ((sampled_zs - z_mu)).unsqueeze(-1)
                ).squeeze(-1)
                d2 = (
                    (sampled_zs - sampled_mus).unsqueeze(1)
                    @ torch.diag_embed(1 / sampled_sigmas)
                    @ ((sampled_zs - sampled_mus)).unsqueeze(-1)
                ).squeeze(-1)

                d = torch.cat((d1, d2), dim=1)
                d_softmax = F.softmax(d / torch.max(d), dim=1) * 0.8

                num_classes = self.args.num_classes
                soften_labels = torch.zeros((batch_size, num_classes))
                soften_labels.fill_(0.2 / (num_classes - 2))

                soften_labels[torch.arange(batch_size), labels.view(-1)] = d_softmax[
                    :, 0
                ]
                soften_labels[
                    torch.arange(batch_size), rand_labels.view(-1)
                ] = d_softmax[:, 1]

                logits2 = model.classifier(sampled_zs)

                loss = self.criterion(logits, labels) + self.criterion(
                    logits2, soften_labels
                )
                loss.backward()

        for name, param in model.named_parameters():
            if name not in self.local_model.classifier_weight_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

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
                r_sigma_softplus = F.softplus(self.r.sigma)

                reg_L2R = z.norm(dim=1).mean() + r_sigma_softplus.norm(dim=2).mean()
                loss += self.l2r_coeff * reg_L2R

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

        self.update_local_dist()

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
