from argparse import Namespace
import copy
import math
from typing import List
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
import numpy as np
import torch.nn.functional as F

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from data_loader import AUG_MAP
from models.base import FedModel
import matplotlib.pyplot as plt
from models.generator import Generator
from tools import (
    aggregate_weights,
)


def sin_growth(alpha, epoch, max_epoch):
    return alpha * math.sin(math.pi / 2 * epoch / max_epoch)


class FedTSGenServer(FedServerBase):
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
        self.ta_clients = args.ta_clients
        self.teacher_clients = args.teacher_clients
        self.alpha = args.alpha
        self.max_round = args.epochs
        self.kpca = KernelPCA(n_components=2)
        self.generator = Generator(
            num_classes=args.num_classes, z_dim=args.z_dim, dataset=args.dataset
        ).to(args.device)

        for client in clients:
            client.generator = self.generator

        self.augment_teacher()

        self.unique_labels = args.num_classes
        self.selected_clients: List[FedTSGenClient] = []

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
        self.ensemble_beta = 1
        self.ensemble_eta = 1

    def compute_avg_client_label_cnts(self):
        num_labels = self.args.num_classes
        label_avg_cnts = sum(
            [
                sum(
                    [
                        self.clients[t_idx].label_cnts[label]
                        for t_idx in self.teacher_clients
                    ]
                )
                for label in range(num_labels)
            ]
        ) // (len(self.teacher_clients) * num_labels)
        return label_avg_cnts

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for client in self.selected_clients:
                weights.append(client.label_cnts[label])
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, epoches=1, n_teacher_iters=5):
        print("Training generator...", end="")
        self.generator.train()
        self.global_model.eval()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        selected_teachers = [
            client
            for client in self.selected_clients
            if client.idx in self.teacher_clients
        ]
        for _ in range(epoches):
            for _ in range(n_teacher_iters):
                self.generator.zero_grad()
                y = np.random.choice(self.qualified_labels, self.args.local_bs)
                y_input = torch.LongTensor(y).to(self.device)
                gen_output, eps = self.generator(y_input)
                # diversity_loss = self.generator.diversity_loss(eps, gen_output)
                ######### get teacher loss ############
                teacher_loss = 0
                teacher_logit = 0
                for client_idx, client in enumerate(selected_teachers):
                    client.local_model.eval()
                    weight = self.label_weights[y][:, client_idx].reshape(-1, 1)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    output = client.local_model.classifier(gen_output)
                    client_output = output.clone().detach()
                    teacher_loss_ = torch.mean(
                        self.generator.crossentropy_loss(client_output, y_input)
                        * torch.tensor(weight, dtype=torch.float32, device=self.device)
                    )
                    teacher_loss += teacher_loss_
                    teacher_logit += client_output * torch.tensor(
                        expand_weight, dtype=torch.float32, device=self.device
                    )
                ######### get student loss ############
                # student_output = self.global_model.classifier(gen_output)
                # student_loss = F.kl_div(
                #     student_output,
                #     teacher_logit, dim=1,
                # )
                loss = (
                    self.ensemble_alpha
                    * teacher_loss
                    # + self.ensemble_beta * student_loss
                    # + self.ensemble_eta * diversity_loss
                )
                loss.backward()
                self.generative_optimizer.step()
        print("done.")

    def augment_teacher(self):
        if len(self.teacher_clients) == 0:
            return

        num_labels = self.args.num_classes
        aug_percent = self.args.augment_percent
        label_avg_cnts = self.compute_avg_client_label_cnts()

        print("before augment:", label_avg_cnts)

        for t_idx in self.teacher_clients:
            teacher: FedTSGenClient = self.clients[t_idx]
            labels_to_aug = []
            labels_aug_num = []
            for label in range(num_labels):
                my_labels = teacher.label_cnts[label]
                if my_labels < label_avg_cnts:
                    to_aug = min(
                        label_avg_cnts - my_labels, int(my_labels * aug_percent)
                    )
                    labels_to_aug.append(label)
                    labels_aug_num.append(to_aug)

            print(labels_to_aug, labels_aug_num)
            teacher.train_loader.dataset.do_augment(
                labels_to_aug, labels_aug_num, AUG_MAP[self.args.dataset]
            )
            teacher.label_cnts = teacher.label_distribution()

        label_avg_cnts = self.compute_avg_client_label_cnts()
        print("after augment:", label_avg_cnts)

    def aggregate_weights(self, weights_map, agg_weights_map):
        if len(weights_map) == 0:
            return None
        agg_weights = np.array([agg_weights_map[idx] for idx in agg_weights_map])
        weights = [weights_map[idx] for idx in weights_map]
        agg_weights /= sum(agg_weights)
        return aggregate_weights(weights, agg_weights, self.client_aggregatable_weights)

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedTSGen Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        global_weight = self.global_weight

        local_losses = []
        local_acc1s = []
        local_acc2s = []
        local_weights = []
        local_protos = []
        local_agg_weights = []

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}
        label_sizes = []
        self.selected_clients = []
        for idx in idx_clients:
            local_client: FedTSGenClient = self.clients[idx]
            self.selected_clients.append(local_client)
            local_epoch = self.args.local_epoch
            local_client.update_local_model(global_weight=global_weight)
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_loss = result.loss_map["round_loss"]
            local_acc1 = result.acc_map["acc1"]
            local_acc2 = result.acc_map["acc2"]

            local_losses.append(local_loss)
            local_acc1s.append(local_acc1)
            local_acc2s.append(local_acc2)

            acc1_dict[f"client_{idx}"] = local_acc1
            acc2_dict[f"client_{idx}"] = local_acc2
            loss_dict[f"client_{idx}"] = local_loss

            agg_weight = local_client.agg_weight()
            weights = copy.deepcopy(w)
            protos = result.protos
            label_cnts = local_client.label_cnts

            local_agg_weights.append(agg_weight)
            local_weights.append(weights)
            local_protos.append(protos)
            label_sizes.append(label_cnts)

        self.global_weight = aggregate_weights(
            local_weights, local_agg_weights, self.client_aggregatable_weights
        )

        self.train_generator()

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

        if self.args.model_het:
            self.analyze_hm_losses(
                idx_clients,
                local_losses,
                local_acc1s,
                local_acc2s,
                result,
                self.args.ta_clients,
                self.args.teacher_clients,
            )

        if self.writer is not None:
            self.writer.add_scalars("clients_acc1", acc1_dict, round)
            self.writer.add_scalars("clients_acc2", acc2_dict, round)
            self.writer.add_scalars("clients_loss", loss_dict, round)
            self.writer.add_scalars("server_acc_avg", result.acc_map, round)
            self.writer.add_scalar("server_loss_avg", loss_avg, round)
        return result


class FedTSGenClient(FedClientBase):
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
        self.generator = None
        self.mse_loss = torch.nn.MSELoss()
        self.label_cnts = self.label_distribution()
        self.available_labels = self.args.num_classes
        label_cnts_list = [
            self.label_cnts[label] for label in range(self.available_labels)
        ]
        avg_label_cnts = int(np.mean(label_cnts_list).item())
        self.unqualified_labels = [
            label
            for label in range(self.available_labels)
            if self.label_cnts[label] < avg_label_cnts
        ]
        self.gen_batch_size = self.args.gen_batch_size
        print(f"client {self.idx} label distribution: {self.label_cnts}")
        print(f"client {self.idx} unqualified labels: {self.unqualified_labels}")
        self.is_attacker = self.idx in args.attackers

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

        for _ in range(local_epoch):
            data_loader = iter(self.train_loader)
            iter_num = len(data_loader)
            if self.args.iter_num > 0:
                iter_num = min(iter_num, self.args.iter_num)
            for _ in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                protos, output = model(images)
                loss0 = self.criterion(output, labels)

                y_input = torch.LongTensor(labels).to(self.device)
                gen_protos, _ = self.generator(y_input)

                # latent presentation loss
                loss1 = self.mse_loss(protos, gen_protos)

                # classifier loss
                sampled_y = np.random.choice(
                    self.unqualified_labels, self.gen_batch_size
                )
                sampled_y = torch.tensor(
                    sampled_y, device=self.device, dtype=torch.int64
                )
                gen_output, _ = self.generator(sampled_y)
                output = self.local_model.classifier(gen_output)
                loss2 = torch.mean(self.generator.crossentropy_loss(output, sampled_y))
                gen_ratio = self.gen_batch_size / self.args.local_bs

                loss3 = protos.norm(dim=1).mean()
                loss = (
                    loss0
                    + self.args.lam * loss1
                    + gen_ratio * loss2
                    + self.args.l2r_coeff * loss3
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

        return result