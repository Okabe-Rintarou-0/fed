from abc import abstractmethod
from tensorboardX import SummaryWriter
from argparse import Namespace
from typing import Any, Dict, List

import torch
from algorithmn.models import GlobalTrainResult, LocalTrainResult
from torch.utils.data import DataLoader
from torch import nn
from models.base import FedModel

from tools import calc_label_distribution


class FedClientBase:
    @abstractmethod
    def __init__(
        self,
        idx: int,
        args: Namespace,
        train_loader: DataLoader,
        test_loader: DataLoader,
        local_model: FedModel,
        writer: SummaryWriter | None,
    ):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_model = local_model
        self.writer = writer
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.global_protos = None

    @abstractmethod
    def label_distribution(self):
        return calc_label_distribution(
            self.train_loader, self.args.num_classes, self.args.get_index
        )

    @abstractmethod
    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        pass

    @abstractmethod
    def clear_memory(self):
        if self.device != "cpu":
            self.local_model = self.local_model.to(torch.device("cpu"))
            torch.cuda.empty_cache()

    @abstractmethod
    def local_test(self) -> float:
        model = self.local_model
        model.eval()
        device = self.args.device
        correct = 0
        total = len(self.test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100.0 * correct / total
        return acc

    @abstractmethod
    def agg_weight(self) -> float:
        data_size = len(self.train_loader.dataset)
        return float(data_size)

    @abstractmethod
    def update_global_protos(self, global_protos):
        self.global_protos = global_protos

    @abstractmethod
    def update_base_model(self, global_weight):
        local_weight = self.local_model.state_dict()
        classifier_weight_keys = self.local_model.classifier_weight_keys
        for k in local_weight.keys():
            if k not in classifier_weight_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    @abstractmethod
    def update_local_classifier(self, new_weight):
        local_weight = self.local_model.state_dict()
        classifier_weight_keys = self.local_model.classifier_weight_keys
        for k in local_weight.keys():
            if k in classifier_weight_keys:
                local_weight[k] = new_weight[k]
        self.local_model.load_state_dict(local_weight)

    @abstractmethod
    def update_local_model(self, global_weight: Dict[str, Any]):
        local_weight = self.local_model.state_dict()
        can_agg_weights = self.local_model.get_aggregatable_weights()
        for k in global_weight.keys():
            if k in can_agg_weights:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)


class FedServerBase:
    @abstractmethod
    def __init__(
        self,
        args: Namespace,
        global_model: FedModel,
        clients: List[FedClientBase],
        writer: SummaryWriter | None,
    ):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.writer = writer

        self.device = args.device

    @staticmethod
    def analyze_hm_losses(
        client_idxs,
        round_losses,
        local_acc1s,
        local_acc2s,
        result: GlobalTrainResult,
        teacher_clients,
    ):
        num_clients = len(client_idxs)
        teacher_losses = []
        teacher_acc1s = []
        teacher_acc2s = []

        student_acc1s = []
        student_acc2s = []
        for i in range(num_clients):
            client_idx = client_idxs[i]
            round_loss = round_losses[i]
            acc1 = local_acc1s[i]
            acc2 = local_acc2s[i]
            if client_idx in teacher_clients:
                teacher_losses.append(round_loss)
                teacher_acc1s.append(acc1)
                teacher_acc2s.append(acc2)
            else:
                student_acc1s.append(acc1)
                student_acc2s.append(acc2)

        if len(teacher_losses) > 0:
            result.loss_map["teacher_avg_loss"] = sum(teacher_losses) / len(
                teacher_losses
            )
            result.acc_map["teacher_acc1"] = sum(teacher_acc1s) / len(teacher_acc1s)
            result.acc_map["teacher_acc2"] = sum(teacher_acc2s) / len(teacher_acc2s)
        
        if len(student_acc1s) > 0:
            result.acc_map["student_acc1"] = sum(student_acc1s) / len(student_acc1s)
            result.acc_map["student_acc2"] = sum(student_acc2s) / len(student_acc2s)

    @abstractmethod
    def train_one_round(self, round: int):
        pass
