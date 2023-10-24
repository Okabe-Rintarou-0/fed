from argparse import Namespace
import copy
from typing import List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
from torch import nn
import numpy as np
import torch.nn.functional as F

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from models.generator import Generator, ProbGenerator
from tools import aggregate_weights


class FedSRGenServer(FedServerBase):
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
        self.generator = ProbGenerator(
            num_classes=args.num_classes, z_dim=args.z_dim, dataset=args.dataset
        )

        for client in clients:
            client.generator = self.generator

        self.batch_size = args.local_bs
        self.unique_labels = args.num_classes
        self.selected_clients: List[FedSRGenClient] = []
        self.device = args.device

        self.generative_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
        )
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98
        )
        self.ensemble_alpha = 1
        self.ensemble_eta = 1

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for client in self.selected_clients:
                weights.append(client.label_counts[label])
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, epoches=1, n_teacher_iters=5):
        self.generator.train()
        self.global_model.eval()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        for _ in range(epoches):
            for _ in range(n_teacher_iters):
                self.generator.zero_grad()
                y = np.random.choice(self.qualified_labels, self.batch_size)
                y_input = torch.LongTensor(y).to(self.device)
                gen_output, _ = self.generator(y_input)
                ######### get teacher loss ############
                teacher_loss = 0
                teacher_logit = 0
                for client_idx, client in enumerate(self.selected_clients):
                    client.local_model.eval()
                    weight = self.label_weights[y][:, client_idx].reshape(-1, 1)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    output = client.local_model.classifier(gen_output)
                    client_output = F.softmax(output, dim=1)
                    teacher_loss_ = torch.mean(
                        self.generator.crossentropy_loss(client_output, y_input)
                        * torch.tensor(weight, dtype=torch.float32)
                    )
                    teacher_loss += teacher_loss_
                    teacher_logit += client_output * torch.tensor(
                        expand_weight, dtype=torch.float32
                    )
                ######### get student loss ############
                # student_output = self.global_model.classifier(gen_output)
                # student_loss = F.kl_div(
                #     F.softmax(student_output, dim=1),
                #     F.softmax(teacher_logit, dim=1),
                # )
                loss = self.ensemble_alpha * teacher_loss
                loss.backward()
                self.generative_optimizer.step()

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedAvg Global Communication Round : {round} ----")
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
            local_client: FedSRGenClient = self.clients[idx]
            self.selected_clients.append(local_client)
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

        # train generator
        self.train_generator()

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


class FedSRGenClient(FedClientBase):
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
        self.label_counts = self.label_distribution()
        self.generator = None
        self.generative_alpha = 10
        self.generative_beta = 10
        self.gen_batch_size = args.gen_batch_size
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.local_bs = args.local_bs
        self.available_labels = [
            i for i, count in enumerate(self.label_distribution()) if count > 0
        ]
        self.l2r_coeff = args.l2r_coeff
        self.cmi_coeff = args.cmi_coeff

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        print(f"[client {self.idx}] local train round {round}:")
        model = self.local_model
        model.train()
        self.generator.eval()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()
        acc1 = self.local_test()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            if self.args.iter_num > 0:
                iter_num = min(iter_num, self.args.iter_num)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                z, output, (z_mu, z_sigma) = model(images, return_dist=True)
                output = F.softmax(output, dim=1)
                predictive_loss = self.criterion(output, labels)

                generative_alpha = self.exp_lr_scheduler(
                    round, decay=0.98, init_lr=self.generative_alpha
                )
                generative_beta = self.exp_lr_scheduler(
                    round, decay=0.98, init_lr=self.generative_beta
                )
                ### get generator output(latent representation) of the same label
                gen_output, (r_mu, r_sigma) = self.generator(labels)
                logit_given_gen = self.local_model.classifier(gen_output)
                target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                latent_loss = generative_beta * self.ensemble_loss(output, target_p)

                ### compute l2 reg
                l2_reg_loss = z.norm(dim=1).mean()

                ### compute cmi reg
                reg_CMI = (
                    torch.log(r_sigma)
                    - torch.log(z_sigma)
                    + (z_sigma**2 + (z_mu - r_mu) ** 2) / (2 * r_sigma**2)
                    - 0.5
                )
                cmi_reg_loss = reg_CMI.sum(1).mean()

                # compute teacher loss
                sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                sampled_y = torch.tensor(sampled_y, device=self.device)
                gen_output, _ = self.generator(sampled_y)
                # latent representation when latent = True, x otherwise
                output = F.softmax(self.local_model.classifier(gen_output), dim=1)
                teacher_loss = generative_alpha * torch.mean(
                    self.generator.crossentropy_loss(output, sampled_y)
                )
                # this is to further balance oversampled down-sampled synthetic data
                gen_ratio = self.gen_batch_size / self.local_bs
                loss = (
                    predictive_loss
                    + gen_ratio * teacher_loss
                    + latent_loss
                    + self.l2r_coeff * l2_reg_loss
                    + self.cmi_coeff * cmi_reg_loss
                )
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
        self.clear_memory()
        return result
