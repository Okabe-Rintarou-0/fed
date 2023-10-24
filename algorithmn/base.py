from abc import abstractmethod
from tensorboardX import SummaryWriter
from argparse import Namespace
from typing import Any, Dict, List

import torch
from algorithmn.models import LocalTrainResult
from torch.utils.data import DataLoader
from torch import nn
from attack import manipulate_one_model
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
        het_model: bool,
        teacher_model: FedModel | None,
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
        self.het_model = het_model
        self.teacher_model = teacher_model

        self.attack = self.idx in args.attackers

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

    @abstractmethod
    def train_one_round(self, round: int):
        pass

    @abstractmethod
    def do_attack(self):
        for client in self.clients:
            if client.attack:
                manipulate_one_model(
                    self.args, client.local_model, client.idx, self.global_model
                )
