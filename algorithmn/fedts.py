from argparse import Namespace
import copy
from typing import List
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from algorithmn.base import FedClientBase, FedServerBase
import numpy as np
import torch.nn.functional as F

from algorithmn.models import GlobalTrainResult, LocalTrainResult
from models.base import FedModel
from models.generator import Generator
from tools import (
    aggregate_weights,
    cal_cosine_difference_vector,
    get_protos,
    optimize_collaborate_vector,
)


class FedTSServer(FedServerBase):
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
        self.alpha = args.alpha
        self.max_round = args.epochs
        self.generator = Generator(
            num_classes=args.num_classes, z_dim=args.z_dim, dataset=args.dataset
        )
        global_model.to(self.device)
        self.initial_global_weights = copy.deepcopy(global_model.state_dict())

        for client in clients:
            client.generator = self.generator

        self.unique_labels = args.num_classes
        self.selected_clients: List[FedTSClient] = []

        self.generative_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
        )

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

    def get_label_weights(self, selected_clients):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for client in selected_clients:
                weights.append(client.label_cnts[label])
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def train_generator(self, round, epoches=1, n_teacher_iters=64):
        print("Training generator...", end="")
        self.generator.train()
        self.global_model.eval()
        self.label_weights, self.qualified_labels = self.get_label_weights(
            self.selected_clients
        )

        local_bs = self.args.local_bs
        for _ in range(epoches):
            for _ in range(n_teacher_iters):
                self.generator.zero_grad()
                y = np.random.choice(self.qualified_labels, local_bs)
                y_input = F.one_hot(torch.LongTensor(y), self.unique_labels).float()
                gen_output, _ = self.generator(y_input)
                teacher_losses = []
                teacher_entropies = []
                for client_idx, client in enumerate(self.selected_clients):
                    client.local_model.eval()
                    weight = self.label_weights[y][:, client_idx].reshape(-1, 1)
                    client_output = client.local_model.classifier(gen_output)

                    logits = F.softmax(client_output, dim=1)
                    entropy = torch.sum(-logits * torch.log(logits + 1e-24), dim=1)

                    teacher_entropies.append(entropy)
                    teacher_loss = (
                        self.generator.crossentropy_loss(client_output, y_input)
                        * torch.tensor(weight, dtype=torch.float32).squeeze()
                    )

                    teacher_losses.append(teacher_loss)

                teacher_losses = torch.vstack(teacher_losses)
                teacher_entropies = torch.vstack(teacher_entropies)
                entropy_weights = F.softmax(1 / teacher_entropies, dim=0)
                ratio = (
                    1.0
                    if client.idx in self.teacher_clients
                    else (round + 1) / self.args.epochs
                )
                losses = torch.mean(
                    teacher_losses * entropy_weights * self.args.eta * ratio, dim=1
                )
                teacher_loss = torch.sum(losses)

                lr2_loss = gen_output.norm(dim=1).mean()
                loss = teacher_loss + self.args.l2r_coeff * lr2_loss
                loss.backward()
                self.generative_optimizer.step()
        print("done.")

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedTS Global Communication Round : {round} ----")
        num_clients = self.args.num_clients
        m = max(int(self.args.frac * num_clients), 1)
        if round >= self.args.epochs:
            m = num_clients
        idx_clients = np.random.choice(range(num_clients), m, replace=False)
        idx_clients = sorted(idx_clients)

        local_losses = []
        local_accs = []
        local_weights = []
        local_agg_weights = []
        local_weights_map = {}

        acc_dict = {}
        loss_dict = {}

        student_weights = []
        student_agg_weights = []

        teacher_weights = []
        teacher_agg_weights = []

        student_accs = []
        student_label_sizes = []

        label_sizes = []
        teacher_accs = []
        teacher_label_sizes = []
        self.selected_clients = []
        for idx in idx_clients:
            local_client: FedTSClient = self.clients[idx]
            self.selected_clients.append(local_client)
            local_epoch = self.args.local_epoch
            result = local_client.local_train(local_epoch=local_epoch, round=round)
            w = result.weights
            local_weights_map[idx] = w
            local_loss = result.loss_map["round_loss"]
            local_acc = result.acc_map["acc"]

            local_losses.append(local_loss)
            local_accs.append(local_acc)

            acc_dict[f"client_{idx}"] = local_acc
            loss_dict[f"client_{idx}"] = local_loss

            agg_weight = local_client.agg_weight()
            weights = copy.deepcopy(w)
            label_cnts = local_client.label_cnts
            local_agg_weights.append(agg_weight)
            local_weights.append(weights)
            label_sizes.append(label_cnts)

            if idx in self.teacher_clients:
                teacher_weights.append(w)
                teacher_agg_weights.append(agg_weight)
                teacher_accs.append(local_acc)
                teacher_label_sizes.append(label_cnts)
            else:
                student_weights.append(w)
                student_agg_weights.append(agg_weight)
                student_accs.append(local_acc)
                student_label_sizes.append(label_cnts)

        student_weights = aggregate_weights(student_weights, student_agg_weights)
        teacher_weights = aggregate_weights(teacher_weights, teacher_agg_weights)
        dv = cal_cosine_difference_vector(
            idx_clients,
            self.initial_global_weights,
            local_weights_map,
        )
        alpha = self.alpha
        sum_local_agg_weights = sum(local_agg_weights)
        for i in range(len(local_agg_weights)):
            local_agg_weights[i] /= sum_local_agg_weights
        agg_weight = optimize_collaborate_vector(dv, alpha, local_agg_weights)
        tmp = torch.tensor(local_agg_weights)
        print(agg_weight, (tmp / torch.sum(tmp)).tolist())

        classifier_weights = aggregate_weights(
            local_weights, agg_weight, self.client_aggregatable_weights
        )

        for local_client in self.clients:
            if local_client.idx in self.teacher_clients:
                local_client.update_base_model(teacher_weights)
            else:
                local_client.update_base_model(student_weights)

            local_client.update_local_classifier(classifier_weights)

        self.train_generator(round)

        loss_avg = sum(local_losses) / len(local_losses)
        acc_avg = sum(local_accs) / len(local_accs)

        result = GlobalTrainResult(
            loss_map={
                "loss_avg": loss_avg,
            },
            acc_map={
                "acc_avg": acc_avg,
            },
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


class FedTSClient(FedClientBase):
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
        self.gen_batch_size = args.gen_batch_size
        gen_ratio = args.gen_batch_size / args.local_bs
        self.lam1 = args.lam1
        self.lam2 = args.lam2
        self.lam3 = args.lam3 * gen_ratio
        self.lam4 = args.lam4

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
        model = self.local_model.to(self.device)
        model.train()
        model.zero_grad()
        round_losses = []
        result = LocalTrainResult()

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
                loss1 = self.criterion(output, labels)
                loss2 = loss3 = loss4 = 0
                self.generator.to(self.device)
                if round > 0:
                    gen_protos, _ = self.generator(labels)

                    # latent presentation loss
                    loss2 = self.mse_loss(protos, gen_protos)

                    # classifier loss
                    sampled_y = torch.tensor(
                        np.random.choice(
                            list(range(self.available_labels)), self.gen_batch_size
                        ),
                        device=self.device,
                    )
                    gen_output, _ = self.generator(sampled_y)
                    output = self.local_model.classifier(gen_output)
                    loss3 = torch.mean(
                        self.generator.crossentropy_loss(output, sampled_y)
                    )

                loss4 = protos.norm(dim=1).mean()
                loss = (
                    self.lam1 * loss1
                    + self.lam2 * loss2
                    + self.lam3 * loss3
                    + self.lam4 * loss4
                )
                loss.backward()
                optimizer.step()
                round_losses.append(loss.item())

        acc = self.local_test()

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

        self.clear_memory()
        self.generator.to("cpu")
        return result
