from typing import List

import torch
from models.base import FedModel
from torch import nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch.nn.functional as F
import torch.distributions as distributions


class ResNetBase(FedModel):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        backbone="resnet18",
        model_het=False,
        z_dim=128,
        input_channel=3,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.num_samples = num_samples
        self.model_het = model_het
        self.z_dim = z_dim

        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError()

        if input_channel == 1:
            self.backbone.conv1 = torch.nn.Conv2d(
                1, 64, (7, 7), (2, 2), (3, 3), bias=False
            )

        out_dim = z_dim * 2 if probabilistic else z_dim
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)
        self.cls = nn.Linear(z_dim, num_classes)
        self.classifier_weight_keys = [
            "cls.weight",
            "cls.bias",
        ]
        self.all_keys = list(self.state_dict().keys())

    def forward(self, x, return_dist=False):
        # --------- Extract Features --------- #
        x = self.backbone(x)

        if not self.probabilistic:
            z = F.leaky_relu(x)
        else:
            z_params = x
            z_mu = z_params[:, : self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim :])
            z_dist = distributions.Independent(
                distributions.normal.Normal(z_mu, z_sigma), 1
            )
            z = z_dist.rsample([self.num_samples]).view([-1, self.z_dim])

        # --------- Classifier --------- #
        y = self.classifier(z)
        if self.probabilistic and return_dist:
            return z, y, (z_mu, z_sigma)
        return z, y

    def classifier(self, z):
        y = self.cls(z)
        y = F.softmax(y, dim=1)
        return y

    def get_aggregatable_weights(self) -> List[str]:
        if not self.model_het:
            return self.all_keys
        # in this case, only classfier can be shared
        return self.classifier_weight_keys


class CifarResNet(ResNetBase):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        backbone="resnet18",
        model_het=False,
        z_dim=128,
    ):
        super().__init__(
            num_classes, probabilistic, num_samples, backbone, model_het, z_dim
        )


class MNISTResNet(ResNetBase):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        backbone="resnet18",
        model_het=False,
        z_dim=128,
    ):
        super().__init__(
            num_classes, probabilistic, num_samples, backbone, model_het, z_dim, 1
        )


class RMNISTResNet(ResNetBase):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        backbone="resnet18",
        model_het=False,
        z_dim=128,
    ):
        super().__init__(
            num_classes, probabilistic, num_samples, backbone, model_het, z_dim
        )


class PACSResNet(ResNetBase):
    def __init__(
        self,
        num_classes=7,
        probabilistic=False,
        num_samples=1,
        backbone="resnet18",
        model_het=False,
        z_dim=128,
    ):
        super().__init__(
            num_classes, probabilistic, num_samples, backbone, model_het, z_dim
        )
