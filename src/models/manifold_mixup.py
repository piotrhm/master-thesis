import hydra
import transformers

import torch
import numpy as np
from torch import nn

import composer.functional as cf

from typing import Any, Optional
from lightning_transformers.task.nlp.text_classification import TextClassificationTransformer
from torch.nn import CrossEntropyLoss


def _gen_mixing_coef(alpha: float) -> float:
    """Samples ``max(z, 1-z), z ~ Beta(alpha, alpha)``."""
    # First check if alpha is positive.
    assert alpha >= 0
    # Draw the mixing parameter from a beta distribution.
    # Check here is needed because beta distribution requires alpha > 0
    # but alpha = 0 is fine for mixup.
    if alpha == 0:
        mixing_lambda = 0
    else:
        mixing_lambda = np.random.beta(alpha, alpha)
    # for symmetric beta distribution, can always use 0 <= lambda <= .5;
    # this way the "main" label is always the original one, which keeps
    # the training accuracy meaningful
    return min(mixing_lambda, 1. - mixing_lambda)



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
    #     distilbert_output = self.model.

    # def forward(
    #         self,
    #         x: torch.Tensor,
    #         attn_mask: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         output_attentions: bool = False,
    #         output_hidden_states: bool = False,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
    #     """
    #     Parameters:
    #         x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
    #         attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.
    #     Returns:
    #         hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
    #         layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
    #             Tuple of length n_layers with the hidden states from each layer.
    #             Optional: only if output_hidden_states=True
    #         all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
    #             Tuple of length n_layers with the attention weights from each layer
    #             Optional: only if output_attentions=True
    #     """
    #     all_hidden_states = () if output_hidden_states else None
    #     all_attentions = () if output_attentions else None
    #
    #     hidden_state = x
    #     for i, layer_module in enumerate(self.layer):
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_state,)
    #
    #         layer_outputs = layer_module(
    #             x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
    #         )
    #         hidden_state = layer_outputs[-1]
    #
    #         if output_attentions:
    #             assert len(layer_outputs) == 2
    #             attentions = layer_outputs[0]
    #             all_attentions = all_attentions + (attentions,)
    #         else:
    #             assert len(layer_outputs) == 1
    #
    #     # Add last layer
    #     if output_hidden_states:
    #         all_hidden_states = all_hidden_states + (hidden_state,)
    #
    #     if not return_dict:
    #         return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
    #     return BaseModelOutput(
    #         last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
    #     )

    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         input_shape = input_ids.size()
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")
    #
    #     device = input_ids.device if input_ids is not None else inputs_embeds.device
    #
    #     if attention_mask is None:
    #         attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
    #
    #     # Prepare head mask if needed
    #     head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    #
    #     if inputs_embeds is None:
    #         inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
    #     return self.transformer(
    #         x=inputs_embeds,
    #         attn_mask=attention_mask,
    #         head_mask=head_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    def training_step_mixup_last_layer(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        if self.pretrained_model_name_or_path == "distilbert-base-uncased":
            permuted_idx = torch.randperm(batch['input_ids'].shape[0])
            x_permuted = batch['input_ids'][permuted_idx]
            y_perm = batch['labels'][permuted_idx]
            mixing = _gen_mixing_coef(self.hparams.alpha)
        else:
            raise NotImplementedError("training step is not implemented for model")

        assert self.model.config.problem_type == self.model.config.problem_type

        output_1 = self.forward_without_classifier(**batch)
        batch['input_ids'] = x_permuted
        output_2 = self.forward_without_classifier(**batch)
        output = (1 - mixing) * output_1 + mixing * output_2

        y_hat = self.model.classifier(output)  # (bs, num_labels) | logits
        loss = self.calculate_mixup_loss(y_hat, batch['labels'], y_perm, mixing)
        preds = torch.argmax(y_hat, dim=1)
        metric_dict = self.compute_metrics(preds, batch["labels"], mode="train")
        self.log_dict(metric_dict, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        # return self.forward_to_layer(batch, batch_idx, dataloader_idx)
        #
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
