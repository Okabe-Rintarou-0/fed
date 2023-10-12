from abc import abstractmethod
from tensorboardX import SummaryWriter
from argparse import Namespace
from typing import Any, Dict, List

import torch
from algorithmn.models import LocalTrainResult
from torch.utils.data import DataLoader
from torch import nn
import numpy as np


class FedClientBase:
    @abstractmethod
    def __init__(self, idx: int, args: Namespace, train_loader: DataLoader, test_loader: DataLoader, local_model: nn.Module, writer: SummaryWriter | None):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_model = local_model
        self.writer = writer
        self.device = args.device

    @abstractmethod
    def local_train(self, local_epoch: int, round: int) -> LocalTrainResult:
        pass

    @abstractmethod
    def local_test(self) -> float:
        pass

    @abstractmethod
    def agg_weight(self) -> torch.tensor:
        data_size = len(self.train_loader.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w

    @abstractmethod
    def update_local_model(self, global_weight: Dict[str, Any]):
        self.local_model.load_state_dict(global_weight)

class FedServerBase:
    @abstractmethod
    def __init__(self, args: Namespace, global_model: nn.Module, clients: List[FedClientBase], writer: SummaryWriter | None):
        self.args = args
        self.global_model = global_model
        self.clients = clients
        self.writer = writer

    @abstractmethod
    def train_one_round(self, round: int):
        pass
        

