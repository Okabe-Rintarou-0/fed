import os
from PIL import ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil
import git

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm


class CloneProgress(git.RemoteProgress):
    def __init__(self, repo: str):
        super().__init__()
        self.pbar = tqdm(desc=f"Clonning {repo}")

    def update(self, op_code, cur_count, max_count=None, message=""):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


class MultipleDomainDataset:
    EPOCHS = 100  # Default, if train with epochs, check performance every epoch.
    N_WORKERS = 4  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.augment_transform = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.datasets = []
        for i, environment in enumerate(environments):
            if augment and (i not in test_envs):
                env_transform = self.augment_transform
            else:
                env_transform = self.transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (
            3,
            224,
            224,
        )
        self.num_classes = len(self.datasets[-1].classes)


class PACS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["art_painting", "cartoon", "photo", "sketch"]

    def download_dataset(self):
        git.Repo.clone_from(
            "https://github.com/MachineLearning2020/Homework3-PACS",
            "./Homework3-PACS",
            progress=CloneProgress("github.com/MachineLearning2020/Homework3-PACS"),
        )
        shutil.move("./Homework3-PACS/PACS", self.dir)
        shutil.rmtree("./Homework3-PACS")

    def __init__(self, root, test_envs, augment=True, download=True):
        self.dir = os.path.join(root, "PACS/")
        if download and not os.path.exists(self.dir):
            self.download_dataset()

        super().__init__(self.dir, test_envs, augment)


class DomainNet(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, augment=True):
        self.dir = os.path.join(root, "domainnet/")
        super().__init__(self.dir, test_envs, augment)


class OfficeHome(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, augment=True):
        self.dir = os.path.join(root, "officehome/")
        super().__init__(self.dir, test_envs, augment)


if __name__ == "__main__":
    pacs = PACS(root="./data", test_envs=[0], augment=False)
    print(pacs[0])
