from typing import List
import torch.nn as nn
from models.base import FedModel


class CifarCNN(FedModel):
    def __init__(
        self,
        num_classes=10,
        probabilistic=False,
        num_samples=1,
        model_het=False,
        z_dim=128,
    ):
        super().__init__()
        self.probabilistic = probabilistic
        self.num_samples = num_samples
        self.model_het = model_het
        self.z_dim = z_dim

        self.fe = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, z_dim),
            nn.LeakyReLU(),
        )

        self.cls = nn.Linear(z_dim, num_classes)
        self.classifier_weight_keys = ["cls.weight", "cls.bias"]
        self.all_keys = list(self.state_dict().keys())

    def forward(self, x):
        # --------- Extract Features --------- #
        z = self.featurize(x)
        # --------- Classifier --------- #
        y = self.classifier(z)
        return z, y

    def classifier(self, z):
        return self.cls(z)

    def featurize(self, x):
        return self.fe(x)

    def get_aggregatable_weights(self) -> List[str]:
        if not self.model_het:
            return self.all_keys
        # in this case, only classfier can be shared
        return self.classifier_weight_keys
