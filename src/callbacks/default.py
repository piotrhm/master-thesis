import os
from typing import Any, Optional, Dict

import torch
import uuid
import pandas as pd

from datetime import datetime
from pytorch_lightning import Callback, Trainer, LightningModule
from torchmetrics.classification.accuracy import Accuracy

from src.utils.helpers import load_txt


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


def calc_l2_norm_transformer(trainer, batch) -> float:
    output = trainer.model.model.forward(**batch)
    loss = output[0]
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


class TrackCleanGradientsTransformer(Callback):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        with torch.enable_grad():
            norm = calc_l2_norm_transformer(trainer, batch)
            pl_module.log("train/g_norm", norm, on_step=False, on_epoch=True, prog_bar=True)
        trainer.optimizers[0].zero_grad()

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with torch.enable_grad():
            norm = calc_l2_norm_transformer(trainer, batch)
            pl_module.log("val/g_norm", norm, on_step=False, on_epoch=True, prog_bar=True)
        trainer.optimizers[0].zero_grad()


class TrackRobustness(Callback):
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        data_dir = trainer.datamodule.hparams.data_dir

        cdata_path = os.path.join(data_dir, 'cifar-10-c')
        corruptions = load_txt(os.path.join(cdata_path, 'corruptions.txt'))
        with torch.enable_grad():
            for cname in corruptions:
                accuracy = Accuracy()
                cdata = trainer.datamodule.ctest_subset_dataloader(cname)
                for batch in cdata:
                    batch[0] = batch[0].to(device=pl_module.device)
                    batch[1] = batch[1].to(device=pl_module.device)
                    loss, preds, targets = pl_module.step(batch)
                    acc = accuracy(preds.detach().cpu(), targets.detach().cpu())
                    pl_module.log("cdata/" + cname + "_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
                    pl_module.log("cdata/" + cname + "_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        data_dir = trainer.datamodule.hparams.data_dir

        cdata_path = os.path.join(data_dir, 'cifar-10-c')
        corruptions = load_txt(os.path.join(cdata_path, 'corruptions.txt'))
        with torch.enable_grad():
            for cname in corruptions:
                accuracy = Accuracy()
                cdata = trainer.datamodule.ctest_subset_dataloader(cname)
                for batch in cdata:
                    batch[0] = batch[0].to(device=pl_module.device)
                    batch[1] = batch[1].to(device=pl_module.device)
                    loss, preds, targets = pl_module.step(batch)
                    acc = accuracy(preds.detach().cpu(), targets.detach().cpu())
                    pl_module.log("cdata_test/" + cname + "_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
                    pl_module.log("cdata_test/" + cname + "_acc", acc, on_step=False, on_epoch=True, prog_bar=True)


class GenerateEH(Callback):
    def __init__(self):
        super(GenerateEH, self).__init__()
        self.ready = False
        self.file_suffix = None
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.file_suffix is None:
            time_stamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            self.file_suffix = f'{trainer.logger._name}_{time_stamp}.csv'

        data_dir = trainer.datamodule.hparams.data_dir
        with torch.enable_grad():
            data = trainer.datamodule.train_set_subset_dataloader()
            accuracy = Accuracy()
            prediction, targets = [], []
            for batch in data:
                batch[0] = batch[0].to(device=pl_module.device)
                batch[1] = batch[1].to(device=pl_module.device)

                loss, pred, target = pl_module.step(batch)

                prediction.extend(pred.detach().cpu())
                targets.extend(target.detach().cpu())

                acc = accuracy(pred.detach().cpu(), target.detach().cpu())
                pl_module.log("data_eh/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
                pl_module.log("data_eh/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            if not self.ready:
                df = pd.DataFrame(targets)
                df.to_csv(os.path.join(data_dir, self.file_suffix), header=False, index=False)
                self.ready = True

            df = pd.read_csv(os.path.join(data_dir, self.file_suffix), header=None)
            df_tmp = pd.DataFrame(prediction)
            df = pd.concat([df, df_tmp], axis=1)
            df.to_csv(os.path.join(data_dir, self.file_suffix), header=False, index=False)
            
            
class GenerateEH_transformer(Callback):
    def __init__(self):
        super(GenerateEH_transformer, self).__init__()
        self.ready = False
        self.file_suffix = None
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.file_suffix is None:
            time_stamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            self.file_suffix = f'{trainer.logger._name}_{time_stamp}.csv'

        data_dir = trainer.datamodule.hparams.data_dir
        with torch.enable_grad():
            data = trainer.datamodule.train_set_subset_dataloader()
            accuracy = Accuracy()
            prediction, targets = [], []
            
            for batch in data:                
                for key, value in batch.items():
                    batch[key] = batch[key].to(device=pl_module.device)

                outputs = pl_module.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1)
                target = batch["labels"]

                prediction.extend(pred.detach().cpu())
                targets.extend(target.detach().cpu())

                acc = accuracy(pred.detach().cpu(), target.detach().cpu())
                pl_module.log("data_eh/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
                pl_module.log("data_eh/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            if not self.ready:
                path_data_dir = os.path.join(os.getcwd(), data_dir)
                os.mkdir(path_data_dir)
                
                df = pd.DataFrame(targets)
                df.to_csv(os.path.join(data_dir, self.file_suffix), header=False, index=False)
                self.ready = True

            df = pd.read_csv(os.path.join(data_dir, self.file_suffix), header=None)
            df_tmp = pd.DataFrame(prediction)
            df = pd.concat([df, df_tmp], axis=1)
            df.to_csv(os.path.join(data_dir, self.file_suffix), header=False, index=False)

