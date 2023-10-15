from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from models.base import FedModel


class CifarCNN(FedModel):
    def __init__(self, num_classes=10, probabilistic=False, num_samples=1, model_het=False, z_dim=128):
        super(CifarCNN, self).__init__()
        self.probabilistic = probabilistic
        self.num_samples = num_samples
        self.model_het = model_het

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        out_dim = z_dim * 2 if probabilistic else z_dim
        self.fc1 = nn.Linear(64 * 3 * 3, out_dim)
        self.fc2 = nn.Linear(z_dim, num_classes, bias=True)
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias',
            'fc1.weight', 'fc1.bias',
        ]
        self.classifier_weight_keys = [
            'fc2.weight', 'fc2.bias',
        ]

    def forward(self, x):
        # --------- Extract Features --------- #
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        if not self.probabilistic:
            z = F.leaky_relu(x)
        else:
            z_params = x
            z_mu = z_params[:, :128]
            z_sigma = F.softplus(z_params[:, 128:])
            z_dist = distributions.Independent(
                distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([self.num_samples]).view([-1, 128])

        # --------- Classifier --------- #
        y = self.fc2(z)
        if self.probabilistic:
            return z, y, (z_mu, z_sigma)
        return z, y

    def get_aggregatable_weights(self) -> List[str]:
        if not self.model_het:
            return list(self.state_dict().keys())
        # in this case, only classfier can be shared
        return self.classifier_weight_keys


class CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                                 'conv2.weight', 'conv2.bias',
                                 'fc1.weight', 'fc1.bias',]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias',]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y
