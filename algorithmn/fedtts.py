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
from tools import aggregate_weights


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

    def aggregate_weights(self, weights_map, agg_weights_map):
        if len(weights_map) == 0:
            return None
        agg_weights = [agg_weights_map[idx] for idx in agg_weights_map]
        weights = [weights_map[idx] for idx in weights_map]
        agg_weights /= sum(agg_weights)
        return aggregate_weights(weights, agg_weights, self.client_aggregatable_weights)

    def aggregate_group_weights(
        self,
        stu_weights,
        ta_weights,
        teacher_weights,
        stu_acc,
        ta_acc,
        teacher_acc,
        stu_agg,
        ta_agg,
        teacher_agg,
    ):
        new_agg_weights = F.softmax(
            torch(
                [stu_agg * stu_acc, ta_agg * ta_acc, teacher_agg * teacher_acc]
                / (stu_agg + ta_agg + teacher_agg)
            )
        )
        weights = [stu_weights, ta_weights, teacher_weights]
        return aggregate_weights(
            weights, new_agg_weights, self.client_aggregatable_weights
        )

    def train_one_round(self, round: int) -> GlobalTrainResult:
        print(f"\n---- FedAvg Global Communication Round : {round} ----")
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

        acc1_dict = {}
        acc2_dict = {}
        loss_dict = {}

        student_agg_weights_map = {}
        student_weights_map = {}
        student_accs = []
        ta_agg_weights_map = {}
        ta_weights_map = {}
        ta_accs = []
        teacher_agg_weights_map = {}
        teacher_weights_map = {}
        teacher_accs = []

        for idx in idx_clients:
            local_client: FedClientBase = self.clients[idx]
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

            if idx in self.ta_clients:
                ta_agg_weights_map[idx] = agg_weight
                ta_weights_map[idx] = weights
                ta_accs.append(local_acc2)
            elif idx in self.teacher_clients:
                teacher_agg_weights_map[idx] = agg_weight
                teacher_weights_map[idx] = weights
                teacher_accs.append(local_acc2)
            else:
                student_agg_weights_map[idx] = agg_weight
                student_weights_map[idx] = weights
                student_accs.append(local_acc2)

        if self.args.attack:
            self.do_attack()

        # aggregate student
        stu_group_weights = self.aggregate_weights(
            student_weights_map, student_agg_weights_map
        )

        # aggregate ta
        ta_group_weights = self.aggregate_weights(ta_weights_map, ta_agg_weights_map)

        # aggregate teacher
        teacher_group_weights = self.aggregate_weights(
            teacher_weights_map, teacher_agg_weights_map
        )

        sum_stu_agg = sum(
            [student_agg_weights_map[idx] for idx in student_agg_weights_map]
        )
        sum_ta_agg = sum([ta_agg_weights_map[idx] for idx in ta_agg_weights_map])
        sum_teacher_agg = sum(
            [teacher_agg_weights_map[idx] for idx in teacher_agg_weights_map]
        )

        stu_acc = sum(student_accs) / len(student_accs)
        ta_acc = sum(ta_accs) / len(ta_accs)
        teacher_acc = sum(teacher_accs) / len(teacher_accs)

        self.global_weight = self.aggregate_group_weights(
            stu_weights=stu_group_weights,
            ta_weights=ta_group_weights,
            teacher_weights=teacher_group_weights,
            stu_acc=stu_acc,
            ta_acc=ta_acc,
            teacher_acc=teacher_acc,
            stu_agg=sum_stu_agg,
            ta_agg=sum_ta_agg,
            teacher_agg=sum_teacher_agg,
        )

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
                _, output = model(images)
                loss = self.criterion(output, labels)
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
