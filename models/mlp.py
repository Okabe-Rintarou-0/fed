from typing import List
from models.base import FedModel
from torch import nn
import torch.distributions as distributions
import torch.nn.functional as F


class MLPBase(FedModel):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        model_het=False,
        z_dim=128,
        input_dim=28 * 28,
        hidden_dim=128,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.num_samples = num_samples
        self.model_het = model_het
        self.z_dim = z_dim
        self.input_dim = input_dim

        self.fe1 = nn.Linear(input_dim, hidden_dim)
        self.fe2 = nn.Linear(hidden_dim, z_dim)
        self.act = nn.LeakyReLU()

        if probabilistic:
            z_dim *= 2
        self.cls = nn.Linear(z_dim, num_classes)
        self.base_weight_keys = [
            "fe1.weight",
            "fe1.bias",
            "fe2.weight",
            "fe2.bias",
        ]
        self.classifier_weight_keys = [
            "cls.weight",
            "cls.bias",
        ]

        self.all_keys = list(self.state_dict().keys())

    def forward(self, x, return_dist=False):
        # --------- Extract Features --------- #
        x = x.view(-1, self.input_dim)
        x = self.act(self.fe1(x))
        x = self.act(self.fe2(x))

        if not self.probabilistic:
            z = x
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


class MNISTMLP(MLPBase):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        model_het=False,
        z_dim=128,
        input_dim=28 * 28,
        hidden_dim=128,
    ):
        super().__init__(
            num_classes,
            probabilistic,
            num_samples,
            model_het,
            z_dim,
            input_dim,
            hidden_dim,
        )
