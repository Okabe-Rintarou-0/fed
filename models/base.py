from abc import abstractmethod
from typing import List
from torch import nn


class FedModel(nn.Module):
    @abstractmethod
    def get_aggregatable_weights(self) -> List[str]:
        pass

    def classifier(self, z):
        pass

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
