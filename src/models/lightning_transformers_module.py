import hydra
import transformers

from typing import TYPE_CHECKING, Type, Any
from lightning_transformers.task.nlp.text_classification import TextClassificationTransformer


class TextClassificationTransformerWrapper(TextClassificationTransformer):
    def __init__(
            self,
            optimizer: Any,
            lr_scheduler: Any,
            monitor: Any,
            pretrained_model_name_or_path,
            num_labels: int,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels
        )
        self.save_hyperparameters(logger=False)
        self.metrics = {}

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
