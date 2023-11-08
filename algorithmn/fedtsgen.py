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
    cal_cosine_difference_matrix,
    cal_cosine_difference_vector,
    cal_protos_diff_vector,
    get_protos,
    optimize_collaborate_vector,
    weight_flatten,
    weight_flatten_cls,
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

    def get_label_weights(self, selected_teachers):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for client in selected_teachers:
                weights.append(client.label_cnts[label])
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, epoches=1, n_teacher_iters=20):
        print("Training generator...", end="")
        self.generator.train()
        self.global_model.eval()
        selected_teachers = [
            client
            for client in self.selected_clients
            if client.idx in self.teacher_clients
        ]
        self.label_weights, self.qualified_labels = self.get_label_weights(
            selected_teachers
        )

        local_bs = self.args.local_bs
        for _ in range(epoches):
            for _ in range(n_teacher_iters):
                self.generator.zero_grad()
                y = np.random.choice(self.qualified_labels, local_bs)
                # y2 = np.random.choice(self.qualified_labels, local_bs)
                y_input = F.one_hot(
                    torch.LongTensor(y).to(self.device), self.unique_labels
                ).float()
                # y_input2 = F.one_hot(
                #     torch.LongTensor(y2).to(self.device), self.unique_labels
                # ).float()
                # lam = torch.rand(local_bs, 1).to(self.device)
                # mixup = lam * y_input + (1 - lam) * y_input2
                gen_output, _ = self.generator(y_input)
                # gen_output2, _ = self.generator(mixup)
                # diversity_loss = self.generator.diversity_loss(eps, gen_output)
                ######### get teacher loss ############
                teacher_logit = 0

                if self.args.entropy_agg:
                    teacher_losses = []
                    teacher_entropies = []
                    for client_idx, client in enumerate(selected_teachers):
                        client.local_model.eval()
                        weight = self.label_weights[y][:, client_idx].reshape(-1, 1)
                        # weight2 = self.label_weights[y2][:, client_idx].reshape(-1, 1)
                        expand_weight = np.tile(weight, (1, self.unique_labels))
                        client_output = client.local_model.classifier(gen_output)

                        logits = F.softmax(client_output, dim=1)
                        entropy = torch.sum(-logits * torch.log(logits + 1e-24), dim=1)

                        teacher_entropies.append(entropy)
                        # client_output2 = client.local_model.classifier(gen_output2)
                        teacher_loss_ = (
                            self.generator.crossentropy_loss(client_output, y_input)
                            * torch.tensor(
                                weight, dtype=torch.float32, device=self.device
                            ).squeeze()
                        )

                        teacher_losses.append(teacher_loss_)
                        # mixup_teacher_loss_ = torch.mean(
                        #     self.generator.crossentropy_loss(client_output2, mixup)
                        #     * torch.tensor(
                        #         weight * weight2, dtype=torch.float32, device=self.device
                        #     )
                        # )
                        teacher_logit += client_output * torch.tensor(
                            expand_weight, dtype=torch.float32, device=self.device
                        )
                    teacher_losses = torch.vstack(teacher_losses)
                    teacher_entropies = torch.vstack(teacher_entropies)
                    entropy_weights = F.softmax(1 / teacher_entropies, dim=0)
                    losses = torch.mean(
                        teacher_losses * entropy_weights * self.args.eta, dim=1
                    )
                    teacher_loss = torch.sum(losses)
                else:
                    teacher_loss = 0
                    for client_idx, client in enumerate(selected_teachers):
                        client.local_model.eval()
                        weight = self.label_weights[y][:, client_idx].reshape(-1, 1)
                        # weight2 = self.label_weights[y2][:, client_idx].reshape(-1, 1)
                        expand_weight = np.tile(weight, (1, self.unique_labels))
                        client_output = client.local_model.classifier(gen_output)

                        # client_output2 = client.local_model.classifier(gen_output2)
                        teacher_loss_ = torch.mean(
                            self.generator.crossentropy_loss(client_output, y_input)
                            * torch.tensor(
                                weight, dtype=torch.float32, device=self.device
                            ).squeeze()
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
                lr2_loss = gen_output.norm(dim=1).mean()
                loss = (
                    self.ensemble_alpha * teacher_loss
                    + self.args.l2r_coeff * lr2_loss
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
                labels_to_aug, labels_aug_num, AUGMENT_TRANSFORM
            )
            teacher.label_cnts = teacher.label_distribution()

        label_avg_cnts = self.compute_avg_client_label_cnts()
        print("after augment:", label_avg_cnts)

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedTSGen Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        local_losses = []
        local_acc1s = []
        local_acc2s = []
        local_weights = []
        local_protos = []
        local_agg_weights = []
        local_weights_map = {}

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        student_weights = []
        student_agg_weights = []

        teacher_weights = []
        teacher_agg_weights = []

        student_accs = []
        student_protos = []
        student_label_sizes = []

        label_sizes = []
        teacher_accs = []
        teacher_protos = []
        teacher_label_sizes = []
        self.selected_clients = []
        for idx in idx_clients:
            local_client: FedTSGenClient = self.clients[idx]
            self.selected_clients.append(local_client)
            local_epoch = self.args.local_epoch
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_weights_map[idx] = w
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
                teacher_weights.append(w)
                teacher_agg_weights.append(agg_weight)
                teacher_accs.append(local_acc2)
                teacher_protos.append(protos)
                teacher_label_sizes.append(label_cnts)
            else:
                student_weights.append(w)
                student_agg_weights.append(agg_weight)
                student_accs.append(local_acc2)
                student_protos.append(protos)
                student_label_sizes.append(label_cnts)

        if self.args.agg_head:
            teacher_weights = aggregate_weights(
                teacher_weights,
                teacher_agg_weights,
                self.client_aggregatable_weights,
            )

            dv = cal_cosine_difference_vector(
                idx_clients,
                teacher_weights,
                local_weights_map,
            )

            # teacher_protos = aggregate_protos(teacher_protos, teacher_label_sizes)
            # agg_weight = torch.zeros((len(idx_clients),))
            alpha = self.alpha
            # for label in range(self.args.num_classes):
            #     if label not in teacher_protos:
            #         continue

            #     this_protos = [
            #         protos[label] if label in protos else teacher_protos[label]
            #         for protos in local_protos
            #     ]
            #     this_label_sizes = [ls[label] for ls in label_sizes]
            #     sum_label_sizes = sum(this_label_sizes)
            #     for i in range(len(this_label_sizes)):
            #         this_label_sizes[i] /= sum_label_sizes

            #     teacher_proto = teacher_protos[label]
            #     dv = cal_protos_diff_vector(
            #         this_protos, teacher_proto, device=self.device
            #     )
            #     agg_weight += optimize_collaborate_vector(dv, alpha, this_label_sizes)

            # agg_weight /= self.args.num_classes
            sum_local_agg_weights = sum(local_agg_weights)
            for i in range(len(local_agg_weights)):
                local_agg_weights[i] /= sum_local_agg_weights
            agg_weight = optimize_collaborate_vector(dv, alpha, local_agg_weights)
            tmp = torch.tensor(local_agg_weights)
            print(agg_weight, (tmp / torch.sum(tmp)).tolist())

            self.global_weight = aggregate_weights(
                local_weights, agg_weight, self.client_aggregatable_weights
            )
        else:
            classfier_weights = aggregate_weights(
                local_weights, local_agg_weights, self.client_aggregatable_weights
            )
            for local_client in self.clients:
                if local_client.idx in self.teacher_clients:
                    local_client.update_base_model(teacher_weights)
                else:
                    local_client.update_base_model(student_weights)

                local_client.update_local_classifier(classfier_weights)

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

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005
        )
        with torch.no_grad():
            org_weights = (
                weight_flatten(model.state_dict()).detach().clone().to(self.device)
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
                n = self.args.num_classes
                sampled_y = (
                    F.one_hot(
                        torch.tensor(sampled_y, device=self.device), num_classes=n
                    )
                    * (0.8 * n - 1)
                    / (n - 1)
                )
                # soften label
                sampled_y += torch.ones_like(sampled_y) * 0.2 / (n - 1)
                gen_output, _ = self.generator(sampled_y)
                output = self.local_model.classifier(gen_output)
                loss2 = torch.mean(self.generator.crossentropy_loss(output, sampled_y))
                gen_ratio = self.gen_batch_size / self.args.local_bs
                cur_weights = weight_flatten(model.state_dict()).to(self.device)
                # loss3 = protos.norm(dim=1).mean()
                if self.args.with_prox:
                    loss3 = self.mse_loss(cur_weights, org_weights)
                else:
                    loss3 = 0
                loss = (
                    loss0
                    + self.args.lam * loss1
                    + gen_ratio * loss2
                    + self.args.mu * loss3
                    # + self.args.l2r_coeff * loss3
                )
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc2 = self.local_test()

        result.protos = self.get_local_protos()
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
