from abc import ABC
from typing import Optional, Tuple

import torch

from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.datamodules.datamodule import DataModule


class CIFAR10DataModule(DataModule, ABC):
    """
    Example of LightningDataModule for CIFAR10 dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        val_test_split: Tuple[int, int] = (5_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.data_train_subset: Optional[Dataset] = None

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """
        Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CIFAR10(self.hparams.data_dir, train=True, transform=self.transform_train)
            dataset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transform)
            self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

        if not self.data_train_subset:
            self.data_train_subset, _ = random_split(
                dataset=CIFAR10(self.hparams.data_dir, train=True, transform=self.transform_train),
                lengths=[1024, 48976],
                generator=torch.Generator().manual_seed(42),
            )

    def train_set_subset_dataloader(self):
        return DataLoader(
            dataset=self.data_train_subset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )