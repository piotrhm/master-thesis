from typing import Any, List

import torch
import composer.functional as cf

from src.models.base_module import BaseModule
from src.models.components.simple_dense_net import SimpleDenseNet


class MixUpModule(BaseModule):
    """Extension of BaseModule. Mixup from mosaicml support added.

    Read the docs:
        https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        alpha=0.2,
        interpolate_loss=True,
    ):
        super().__init__(
            net=net,
            lr=lr,
            weight_decay=weight_decay
        )
        self.save_hyperparameters(logger=False)

    def step(self, batch: Any):
        x, y = batch
        x, y_perm, mixing = cf.mixup_batch(x, y, alpha=self.hparams.alpha)

        if self.hparams.interpolate_loss:
            y_hat = self.forward(x)
            loss = (1 - mixing) * self.criterion(y_hat, y) + mixing * self.criterion(y_hat, y_perm)
        else:
            y_mixed = (1 - mixing) * y + mixing * y_perm
            y_hat = self.forward(x)
            loss = self.criterion(y_hat, y_mixed)

        preds = torch.argmax(y_hat, dim=1)
        return loss, preds, y


