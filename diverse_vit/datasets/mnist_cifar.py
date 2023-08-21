import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import diverse_vit.datasets.scripts_mnist_cifar.mnistcifar_utils as mc_utils


class MNISTCIFARDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        domain,
        causal_noise=0.0,  # causal is 100% correct
        spurious_ratio=0.5,  # spourious is completly uncorrelated
        transform=None,
        target_transform=None,
    ):
        """Initializes a MNIST-CIFAR dataset.

        Args:
            split (str): Specifies the dataset split: ``'train'`` | ``'test'``.
            transform (optional): Input transformation. Defaults to None.
            target_transform (optional): Target label transformation. Defaults to None.
        """
        self.data_path = data_path
        self.split = split

        self.transform = transform
        self.target_transform = target_transform

        mnist_classes = (0, 1)
        cifar_classes = (1, 9)
        randomize_mnist = spurious_ratio
        randomize_cifar = causal_noise

        print(f"[{domain}] Using MNIST-CIFAR dataset with MNIST randomness of: {randomize_mnist}")
        (images_train, targets_train), (images_test, targets_test) = mc_utils.get_mnist_cifar(
            data_path=data_path,
            mnist_classes=mnist_classes,
            cifar_classes=cifar_classes,
            randomize_mnist=randomize_mnist,
            randomize_cifar=randomize_cifar,
        )
        if split == "train":
            self.images = images_train
            self.targets = targets_train
        else:
            self.images = images_test
            self.targets = targets_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieve a dataset example.

        Args:
            idx (int): The index of the dataset example to be returned.

        Returns:
            dict: A dictionary containing the image and its attributes in
                the following format:
                {
                    'img' (ndarray)
                    'target' (int)
                }
        """
        img = self.images[idx]
        target = self.targets[idx].astype(np.uint8)

        if self.transform is not None:
            img = self.transform(img.transpose(1, 2, 0))

        if self.target_transform is not None:
            target = self.target_transform(target)

        data_dict = {"img": img, "target": target}

        return data_dict


def main():
    pass


if __name__ == "__main__":
    main()
