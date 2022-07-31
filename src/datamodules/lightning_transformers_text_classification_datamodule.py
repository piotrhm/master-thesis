from lightning_transformers.task.nlp.text_classification import TextClassificationDataModule
from transformers import AutoTokenizer


class TextClassificationDataModuleWrapper(TextClassificationDataModule):
    """Defines the ``LightningDataModule`` for Text Classification Datasets."""

    def __init__(self, tokenizer_pretrained_model_name_or_path, *args, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_model_name_or_path)

        super().__init__(self.tokenizer, *args, **kwargs)
        self.labels = None
