from torch import nn
import torch
import torch.distributions as distributions
import torch.nn.functional as F

GENERATORCONFIGS = {
    # hidden_dimension, input_channel
    "mnist": (512, 28),
    "fmnist": (512, 28),
    "emnist": (512, 28),
    "rmnist": (512, 28),
    "cifar": (512, 32),
    "cinic10": (512, 32),
    "cifar100": (512, 32),
    "pacs": (512, 224),
}


class Generator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_dim: int,
        dataset="cifar",
    ):
        super(Generator, self).__init__()
        self.dataset = dataset
        self.n_class = num_classes
        self.hidden_dim, self.input_channel = GENERATORCONFIGS[dataset]
        self.noise_dim = self.latent_dim = z_dim
        input_dim = self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)

        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        self.representation_layer = nn.Linear(
            self.fc_configs[-1], self.latent_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand(
            (batch_size, self.noise_dim), device=labels.device
        )
        if len(labels.shape) == 1:
            y_input = torch.FloatTensor(
                batch_size, self.n_class).to(labels.device)
            y_input.zero_()
            y_input.scatter_(1, labels.view(-1, 1), 1)
        else:
            y_input = labels
        z = torch.cat((eps, y_input), dim=1)
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        z = F.leaky_relu(z)
        return z, eps


class ProbGenerator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_dim: int,
        dataset="cifar",
    ):
        super(ProbGenerator, self).__init__()
        self.dataset = dataset
        self.n_class = num_classes
        self.hidden_dim, self.input_channel = GENERATORCONFIGS[dataset]
        self.latent_dim = z_dim * 2
        self.num_samples = 1
        self.z_dim = z_dim
        input_dim = self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)

        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        self.representation_layer = nn.Linear(
            self.fc_configs[-1], self.latent_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        if len(labels.shape) == 1:
            y_input = torch.FloatTensor(
                batch_size, self.n_class).to(labels.device)
            y_input.zero_()
            y_input.scatter_(1, labels.view(-1, 1), 1)
        else:
            y_input = labels
        z = y_input
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        z_mu = z[:, : self.z_dim]
        z_sigma = F.softplus(z[:, self.z_dim:])
        z_dist = distributions.Independent(
            distributions.normal.Normal(z_mu, z_sigma), 1
        )
        z = z_dist.rsample([self.num_samples]).view([-1, self.z_dim])

        return z, (z_mu, z_sigma)
