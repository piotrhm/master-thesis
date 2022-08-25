from abc import ABC
from typing import Optional, Tuple

import torch

from torch.utils.data import ConcatDataset, random_split
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms

from src.datamodules.datamodule import DataModule


class CIFAR100DataModule(DataModule, ABC):
    """
    Example of LightningDataModule for CIFAR100 dataset.
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
            val_test_split=val_test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
        ])
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
        ])

    @property
    def num_classes(self) -> int:
        return 100

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        CIFAR100(self.hparams.data_dir, train=True, download=True)
        CIFAR100(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CIFAR100(self.hparams.data_dir, train=True, transform=self.transform_train)
            dataset = CIFAR100(self.hparams.data_dir, train=False, transform=self.transform)
            self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
