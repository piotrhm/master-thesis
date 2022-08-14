import hydra
import transformers
import torch
import composer.functional as cf

from typing import Any
from lightning_transformers.task.nlp.text_classification import TextClassificationTransformer
from torch.nn import CrossEntropyLoss


class TextClassificationTransformerWrapperMixup(TextClassificationTransformer):
    def __init__(
            self,
            optimizer: Any,
            lr_scheduler: Any,
            monitor: Any,
            pretrained_model_name_or_path,
            num_labels: int,
            alpha=1,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels
        )
        self.save_hyperparameters(logger=False)
        self.metrics = {}
        self.criterion = CrossEntropyLoss()

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        batch_new = batch.copy()
        if self.pretrained_model_name_or_path == "distilbert-base-uncased":
            batch_new['inputs_embeds'] = self.model.distilbert.embeddings(batch_new['input_ids'])
            batch_new.pop('input_ids', None)
        else:
            raise NotImplementedError("training step is not implemented for model")

        x, y_perm, mixing = cf.mixup_batch(batch_new['inputs_embeds'], batch_new['labels'], alpha=self.hparams.alpha)
        batch_new.pop('inputs_embeds', None)
        batch_new['inputs_embeds'] = x

        assert self.model.config.problem_type == self.model.config.problem_type

        outputs = self.model(**batch_new)
        y_hat = outputs[1]
        loss = (1 - mixing) * self.criterion(y_hat.view(-1, self.model.num_labels), batch['labels'].view(-1)) +\
               mixing * self.criterion(y_hat.view(-1, self.model.num_labels), y_perm.view(-1))

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        metric_dict = self.compute_metrics(preds, batch["labels"], mode="train")
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        config_opt = self.hparams.optimizer
        config_opt.pop('_partial_')
        opt = hydra.utils.instantiate(config_opt, params=self.parameters())

        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)

        if "monitor" in self.hparams:
            scheduler = {'scheduler': scheduler, 'monitor': self.hparams.monitor.metric_to_track}

        return [opt], [scheduler]
