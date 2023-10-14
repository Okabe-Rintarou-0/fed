from typing import List
from models.base import FedModel
from torch import nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch.nn.functional as F
import torch.distributions as distributions


class CifarResnet(FedModel):
    def __init__(self, num_classes=10, probabilistic=False, num_samples=1, backbone='resnet18', model_het=False):
        super().__init__()
        self.probabilistic = probabilistic
        self.num_samples = num_samples
        self.model_het = model_het

        if backbone == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError()

        out_dim = 256 if probabilistic else 128
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.classifier_weight_keys = [
            'fc2.weight', 'fc2.bias',
        ]

    def forward(self, x):
        # --------- Extract Features --------- #
        x = self.backbone(x)

        if not self.probabilistic:
            z = x
        else:
            z_params = x
            z_mu = z_params[:, :128]
            z_sigma = F.softplus(z_params[:, 128:])
            z_dist = distributions.Independent(
                distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([self.num_samples]).view([-1, 128])

        # --------- Classifier --------- #
        y = self.fc2(z)
        return z, y

    def get_aggregatable_weights(self) -> List[str]:
        if not self.model_het:
            return list(self.state_dict().keys())
        # in this case, only classfier can be shared
        return self.classifier_weight_keys