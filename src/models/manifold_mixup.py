from typing import Any

import torch
import composer.functional as cf

from src.models.base_module import BaseModule


class ManifoldMixUpModule(BaseModule):
    """Extension of BaseModule. Mixup from mosaicml support added.

    Read the docs:
        https://github.com/mosaicml/composer/tree/dev/composer/algorithms/mixup
    """

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        lr_scheduler: Any,
        monitor: Any,
        interpolate_loss=True,
        alpha=0.2,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            monitor=monitor
        )
        self.save_hyperparameters(logger=False)

    def step_mixup(self, batch: Any):
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

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step_mixup(batch)
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}