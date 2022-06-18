from pytorch_lightning import Callback, Trainer, LightningModule
from typing import Any, Optional, Dict
import torch


def calc_l2_norm(trainer, x, y) -> float:
    logits = trainer.model.forward(x)
    loss = trainer.model.criterion(logits, y)
    loss.backward()
    total_norm = 0
    parameters = [p for p in trainer.model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class TrackCleanGradients(Callback):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        x, y = batch
        with torch.enable_grad():
            norm = calc_l2_norm(trainer, x, y)
            pl_module.log("train/g_norm", norm, on_step=False, on_epoch=True, prog_bar=True)
        trainer.optimizers[0].zero_grad()

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        x, y = batch
        with torch.enable_grad():
            norm = calc_l2_norm(trainer, x, y)
            pl_module.log("val/g_norm", norm, on_step=False, on_epoch=True, prog_bar=True)
        trainer.optimizers[0].zero_grad()
