import torchvision
import matplotlib.pyplot as plt
import numpy as np


def show_img_batch(imgs):
    imgs = torchvision.utils.make_grid(imgs).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.show()
