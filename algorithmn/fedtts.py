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
from data_loader import AUGMENT_TRANSFORM
from models.base import FedModel
import matplotlib.pyplot as plt
from models.generator import Generator
from tools import (
    aggregate_protos,
    aggregate_weights,
    cal_protos_diff_vector,
    get_protos,
    optimize_collaborate_vector,
)


def sin_growth(alpha, epoch, max_epoch):
    return alpha * math.sin(math.pi / 2 * epoch / max_epoch)


class FedTTSServer(FedServerBase):
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

        self.augment_teacher()

        self.unique_labels = args.num_classes
        self.selected_clients: List[FedTTSClient] = []

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
                diversity_loss = self.generator.diversity_loss(eps, gen_output)
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
                    self.ensemble_alpha * teacher_loss
                    + self.ensemble_eta * diversity_loss
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
            teacher: FedTTSClient = self.clients[t_idx]
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
                labels_to_aug, labels_aug_num, AUGMENT_TRANSFORM
            )
            teacher.label_cnts = teacher.label_distribution()

        label_avg_cnts = self.compute_avg_client_label_cnts()
        print("after augment:", label_avg_cnts)

    def plot_protos_kpca(self, idx_clients, local_protos, label_sizes):
        for label in range(self.args.num_classes):
            this_protos = [protos[label] for protos in local_protos if label in protos]
            this_idx_clients = [
                idx_clients[idx]
                for (idx, protos) in enumerate(local_protos)
                if label in protos
            ]
            this_label_sizes = [
                label_sizes[idx][label]
                for (idx, protos) in enumerate(local_protos)
                if label in protos
            ]
            sum_this_label_sizes = sum(this_label_sizes)
            for i in range(len(this_label_sizes)):
                this_label_sizes[i] /= sum_this_label_sizes

            this_protos = torch.vstack(this_protos).to("cpu")
            pca = self.kpca.fit_transform(this_protos)
            kmeans = KMeans(n_clusters=1, n_init="auto")
            k = kmeans.fit(pca)
            distances = k.transform(pca)
            avg_distances = np.min(distances, axis=1).mean()

            center = k.cluster_centers_[0]
            circle = plt.Circle(
                (center[0], center[1]), avg_distances, color="blue", alpha=0.3
            )
            plt.gca().add_patch(circle)
            plt.scatter(x=center[0], y=center[1], label="center")
            x = pca[:, 0]
            y = pca[:, 1]

            # for i in range(x.shape[0]):
            #     x[i] = center[0] + (x[i] - center[0]) * this_label_sizes[i]
            #     y[i] = center[1] + (y[i] - center[1]) * this_label_sizes[i]

            benign_clients = list(range(x.shape[0]))
            if len(self.args.attackers) > 0:
                attackers = [
                    i
                    for (i, idx) in enumerate(this_idx_clients)
                    if idx in self.args.attackers
                ]
                benign_clients = list(set(benign_clients) - set(attackers))
                plt.scatter(x[attackers], y[attackers], label="attackers")

            plt.scatter(x[benign_clients], y[benign_clients], label="benign clients")
            plt.legend()
            plt.show()

            # c0 = [idx for (idx, label) in enumerate(k.labels_) if label == 0]
            # c1 = [idx for (idx, label) in enumerate(k.labels_) if label == 1]

            # plt.clf()
            # plt.scatter(x[c0], y[c0], label='c0')
            # plt.scatter(x[c1], y[c1], label='c1')
            # plt.legend()
            # plt.show()

    def aggregate_weights(self, weights_map, agg_weights_map):
        if len(weights_map) == 0:
            return None
        agg_weights = np.array([agg_weights_map[idx] for idx in agg_weights_map])
        weights = [weights_map[idx] for idx in weights_map]
        agg_weights /= sum(agg_weights)
        return aggregate_weights(weights, agg_weights, self.client_aggregatable_weights)

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedTTS Global Communication Round : {round} ----")
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

        student_agg_weights_map = {}
        student_weights_map = {}
        student_accs = []
        student_protos = []
        student_label_sizes = []

        label_sizes = []

        teacher_agg_weights_map = {}
        teacher_weights_map = {}
        teacher_accs = []
        teacher_protos = []
        teacher_label_sizes = []
        self.selected_clients = []
        for idx in idx_clients:
            local_client: FedTTSClient = self.clients[idx]
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

            if idx in self.teacher_clients:
                teacher_agg_weights_map[idx] = agg_weight
                teacher_weights_map[idx] = weights
                teacher_accs.append(local_acc2)
                teacher_protos.append(protos)
                teacher_label_sizes.append(label_cnts)
            else:
                student_agg_weights_map[idx] = agg_weight
                student_weights_map[idx] = weights
                student_accs.append(local_acc2)
                student_protos.append(protos)
                student_label_sizes.append(label_cnts)

        # self.plot_protos_kpca(idx_clients, teacher_protos, label_sizes)

        self.global_weight = aggregate_weights(
            local_weights, local_agg_weights, self.client_aggregatable_weights
        )

        teacher_protos = aggregate_protos(teacher_protos, teacher_label_sizes)

        global_protos = {}

        for label in range(self.args.num_classes):
            if label not in teacher_protos:
                continue

            this_protos = [
                protos[label] if label in protos else teacher_protos[label]
                for protos in local_protos
            ]
            teacher_proto = teacher_protos[label]
            dv = cal_protos_diff_vector(this_protos, teacher_proto, device=self.device)

            for i, client in enumerate(self.selected_clients):
                dv[i] /= client.label_div

            this_label_cnts = [
                this_label_sizes[label] for this_label_sizes in label_sizes
            ]
            label_cnts_sum = sum(this_label_cnts)
            for idx in range(len(this_label_cnts)):
                this_label_cnts[idx] /= label_cnts_sum

            alpha = sin_growth(self.alpha, round, self.max_round)
            agg_weight = optimize_collaborate_vector(dv, alpha, this_label_cnts)
            global_protos[label] = torch.zeros_like(teacher_proto, device=self.device)
            print(agg_weight)
            for idx, protos in enumerate(this_protos):
                global_protos[label] += agg_weight[idx] * protos

        # self.train_generator()

        # update global protos
        for client in self.clients:
            client.update_global_protos(global_protos)

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


class FedTTSClient(FedClientBase):
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
        self.mse_loss = torch.nn.MSELoss()
        self.label_cnts = self.label_distribution()
        self.is_attacker = self.idx in args.attackers

        self.label_div = sum(
            [1 for label in range(args.num_classes) if self.label_cnts[label] > 0]
        )

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

        acc1 = self.local_test()

        local_protos1 = self.get_local_protos()

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )
        global_protos = self.global_protos

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
                optimizer.step()
                round_losses.append(loss.item())

        acc2 = self.local_test()
        local_protos2 = self.get_local_protos()

        result.weights = model.state_dict()
        result.protos = local_protos2
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
