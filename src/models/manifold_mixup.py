import hydra
import transformers

import torch
from torch import nn

import composer.functional as cf

from typing import Any, Optional
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
            mode='embedding'
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels
        )
        assert mode in ['embedding', 'last_layer']
        self.mode = mode
        self.save_hyperparameters(logger=False)
        self.metrics = {}
        self.criterion = CrossEntropyLoss()

    def calculate_mixup_loss(self, y_hat, y, y_perm, mixing):
        return (1 - mixing) * self.criterion(y_hat.view(-1, self.model.num_labels), y.view(-1)) +\
               mixing * self.criterion(y_hat.view(-1, self.model.num_labels), y_perm.view(-1))

    def training_step_mixup_embedding(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
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

        output = self.model(**batch_new)
        y_hat = output[1]  # logit
        loss = self.calculate_mixup_loss(y_hat, batch['labels'], y_perm, mixing)
        preds = torch.argmax(y_hat, dim=1)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode="train")
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def forward_without_classifier(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        distilbert_output = self.model.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.model.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.model.dropout(pooled_output)  # (bs, dim)

        return pooled_output

    # def forward_to_layer(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ):
    #     distilbert_output = self.model.distilbert(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #
    #     return pooled_output

    def training_step_mixup_last_layer(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if self.pretrained_model_name_or_path == "distilbert-base-uncased":
            permuted_idx = torch.randperm(batch['input_ids'].shape[0])
            x_permuted = batch['input_ids'][permuted_idx]
            batch_perm = batch.copy()
            batch_perm['input_ids'] = x_permuted

            x, y_perm, mixing = cf.mixup_batch(
                input=batch['input_ids'],
                target=batch['labels'],
                alpha=self.hparams.alpha,
                indices=permuted_idx,
            )
        else:
            raise NotImplementedError("training step is not implemented for model")

        assert self.model.config.problem_type == self.model.config.problem_type

        output_1 = self.forward_without_classifier(**batch)
        output_2 = self.forward_without_classifier(**batch_perm)
        output = (1 - mixing) * output_1 + mixing * output_2

        y_hat = self.model.classifier(output)  # (bs, num_labels) | logits
        loss = self.calculate_mixup_loss(y_hat, batch['labels'], y_perm, mixing)
        preds = torch.argmax(y_hat, dim=1)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode="train")
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if self.mode == 'embedding':
            return self.training_step_mixup_embedding(batch, batch_idx, dataloader_idx)
        elif self.mode == 'last_layer':
            return self.training_step_mixup_last_layer(batch, batch_idx, dataloader_idx)
        else:
            raise NotImplementedError("provided mode not implemented")

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
