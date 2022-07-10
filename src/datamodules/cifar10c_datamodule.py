from abc import ABC
from typing import Optional, Tuple

import os
import torch

from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from src.datasets.cifar10c import CIFAR10C, extract_subset
from src.datamodules.cifar10_datamodule import CIFAR10DataModule
from src.utils.helpers import load_txt


class CIFAR10CDataModule(CIFAR10DataModule, ABC):
    """
    LightningDataModule for CIFAR10 dataset with corruptions.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (50_000, 5_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.cdata = {}

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        super().setup(stage)

        cdata_path = os.path.join(self.hparams.data_dir, 'cifar-10-c')
        corruptions = load_txt(os.path.join(cdata_path, 'corruptions.txt'))
        for cname in corruptions:
            dataset = CIFAR10C(
                cdata_path,
                cname,
                transform=self.transform_test,
                target_transform=None
            )
            self.cdata[cname] = dataset

    def ctest_dataloader(self, cname):
        return DataLoader(
            dataset=self.cdata[cname],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def ctest_subset_dataloader(self, cname, num_subset=100, random_subset=True):
        return DataLoader(
            dataset=extract_subset(self.cdata[cname], num_subset, random_subset),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
