import numpy as np
import torch
from data_loader import get_dataloaders, get_model
from options import parse_args
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

args = parse_args()
model = get_model(args)

sd = torch.load("./training_data/global.pth", map_location="cpu")

r_mus = sd["r.mu"]
r_sigmas = sd["r.sigma"]
r_C = sd["r.C"]

sd.pop("r.mu")
sd.pop("r.sigma")
sd.pop("r.C")

model.load_state_dict(sd)
_, test_loaders = get_dataloaders(args)

test_loader = test_loaders[0]

classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

for images, labels in test_loader:
    z, output, (z_mu, z_sigma) = model(images, return_dist=True)
    predict = torch.argmax(output, dim=1)

    for i in range(len(predict)):
        p = predict[i]
        label = labels[i]

        if p != label:
            print(f"ground truth: {classes[label]}, predict: {classes[p]}")
            conf = {
                clazz: f"{output[i][label].item():.2f}"
                for label, clazz in enumerate(classes)
            }
            print(conf)

            image_array = np.transpose(images[i].numpy(), (1, 2, 0))
            image_array = (
                (image_array - image_array.min())
                / (image_array.max() - image_array.min())
                * 255
            ).astype(np.uint8)

            dm = {}
            ds = []
            for label in range(len(classes)):
                r_mu = r_mus[label]
                r_sigma = r_sigmas[label]

                d = (
                    (z[i] - r_mu).unsqueeze(0)
                    @ torch.diag_embed(1 / r_sigma)
                    @ ((z[i] - r_mu)).unsqueeze(-1)
                ).squeeze(-1)
                ds.append(d)

            ds = F.softmax(torch.cat(ds), dim=0)
            for label in range(len(classes)):
                dm[classes[label]] = ds[label].item()

            print(dm)

            image = Image.fromarray(image_array)
            image.show()
            _ = input()
