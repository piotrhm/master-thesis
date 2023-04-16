from lightning_transformers.task.nlp.text_classification import TextClassificationDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from typing import Optional

import torch

class TextClassificationDataModuleWrapper(TextClassificationDataModule):
    """Defines the ``LightningDataModule`` for Text Classification Datasets."""

    def __init__(self, tokenizer_pretrained_model_name_or_path, data_dir: str = "data/", *args, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_model_name_or_path)
        
        super(TextClassificationDataModule, self).__init__(self.tokenizer, *args, **kwargs)
        self.labels = None
        self.data_train_subset = None
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        

    def setup(self, stage: Optional[str] = None):
        super(TextClassificationDataModuleWrapper, self).setup()
        self.data_train_subset, _ = random_split(
            dataset=self.ds['train'],
            lengths=[1024, 14976],
            generator=torch.Generator().manual_seed(42),
        )

    def train_set_subset_dataloader(self):
        return DataLoader(
            dataset=self.data_train_subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
