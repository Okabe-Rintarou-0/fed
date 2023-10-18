from argparse import Namespace
import copy
import math
from typing import List, Tuple
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
from torch import nn
import numpy as np
from torch import distributions

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from tools import aggregate_weights, calculate_matmul, calculate_matmul_n_times

# Reference: https://github.com/zshuai8/FedGMM_ICML2023


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
        self.z_dim = z_dim
        device = args.device
        self.em_iter = args.em_iter
        self.qs = None
        self.mu = torch.randn(1, self.M1, z_dim).to(device)
        self.eps = 1.e-1
        self.var = torch.eye(z_dim).reshape(
            1, 1, z_dim, z_dim).repeat(1, self.M1, 1, 1).to(device)
        self.pi = torch.Tensor(1, self.M1, 1).fill_(1. / self.M1).to(device)

        self.mu.requires_grad = self.var.requires_grad = self.pi.requires_grad = False
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.params_fitted = False

        self.init_gmm()

    def _gather_losses(self) -> torch.Tensor:
        num = len(self.train_loader.dataset)
        losses = np.zeros(num)
        with torch.no_grad():
            for x, labels, index in self.train_loader:
                x = x.to(self.device)
                _, logits = self.local_model(x)
                losses[index] = self.criterion(logits, labels)
        return torch.Tensor(losses)

    def _gather(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num = len(self.train_loader.dataset)
        zs, losses = np.zeros((num, self.z_dim)), np.zeros(num)
        with torch.no_grad():
            for x, labels, index in self.train_loader:
                x = x.to(self.device)
                z, logits = self.local_model(x)
                zs[index] = z
                losses[index] = self.criterion(logits, labels)
        zs = torch.Tensor(zs)
        losses = torch.Tensor(losses)
        return zs, losses

    def init_gmm(self):
        z, losses = self._gather()
        self.mu = self._get_kmeans_mu(z, n_centers=self.M1)

        # compute log(q_s) in E step, in log space, to prevent overflow errors.
        _, log_resp = self._e_step(z, losses)
        _, mu, var = self._m_step(z, log_resp)
        self.__update_mu(mu)
        self.__update_var(var)

    def __em(self):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """
        z, losses = self._gather()
        _, log_resp = self._e_step(z, losses)
        pi, mu, var = self._m_step(z, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

        self.qs = torch.exp(log_resp)

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         torch.FloatTensor
        """
        assert pi.size() in [
            (1, self.M1, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
            1, self.M1, 1)

        self.pi.data = pi

        self.pi = self.pi.to(self.device)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         torch.FloatTensor
        """
        assert mu.size() in [(self.M1, self.z_dim), (1, self.M1, self.z_dim)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
            self.M1, self.z_dim, self.M1, self.z_dim)

        if mu.size() == (self.M1, self.z_dim):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.M1, self.z_dim):
            self.mu = mu

        self.mu = self.mu.to(self.device)

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        torch.FloatTensor
        """
        assert var.size() in [(self.M1, self.z_dim, self.z_dim), (1, self.M1, self.z_dim, self.z_dim)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i), but instead {}".format(
            self.M1, self.z_dim, self.z_dim, self.M1, self.z_dim, self.z_dim, var.size())

        if var.size() == (self.M1, self.z_dim, self.z_dim):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.M1, self.z_dim, self.z_dim):
            self.var = var

        assert not torch.isnan(self.var).any()

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          torch.Tensor (n, d) or (n, 1, d)
            log_resp:   torch.Tensor (n, k, 1)
        returns:
            pi:         torch.Tensor (1, k, 1)
            mu:         torch.Tensor (1, k, d)
            var:        torch.Tensor (1, k, d)
        """
        x = self.check_size(x)

        # resp -> q_s
        resp = torch.exp(log_resp)

        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
        eps = (torch.eye(self.z_dim) * self.eps).to(x.device)
        var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
                        keepdim=True) / pi.unsqueeze(3) + eps
        pi = pi / x.shape[0]

        return pi, mu, var

    def check_size(self, x):
        if len(x.size()) == 2:
            # (n, d) --> (n, 1, d)
            x = x.unsqueeze(1)

        return x

    def _get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
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

        for _ in range(init_times):
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

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = torch.empty(size=(self.M1,)).to(var.device)

        log_det = torch.linalg.slogdet(var)
        log_det = log_det.logabsdet * log_det.sign
        return log_det.unsqueeze(-1)

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
        multivariate_normal = distributions.MultivariateNormal(self.mu, self.var)
        log_prob = multivariate_normal.log_prob(x)
        assert not torch.isnan(log_prob).any()
        return log_prob.unsqueeze(-1)

    def _e_step(self, x, logit_losses: torch.Tensor):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  torch.Tensor (1)
            log_resp:       torch.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        # q ~ pi_c * N(x, mu, var) * p(y|x)
        # shape: [n, k, 1]
        weighted_log_prob = self._estimate_log_prob(x) +\
            torch.log(self.pi) - logit_losses.view(-1, 1, 1)

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm
        return torch.mean(log_prob_norm), log_resp

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
            # EM algorithmn iterations
            for _ in range(self.em_iter):
                self.__em()

            for images, labels, index in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                qs = self.qs[index].squeeze()
                qs = torch.sum(qs, dim=-1)
                loss = self.criterion(output, labels).mean()
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
