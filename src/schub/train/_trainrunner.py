from typing import TYPE_CHECKING, Callable, Optional

from transformers import Trainer, TrainingArguments

from schub.dataloader import DataSplitter, default_data_collator

# Use type checking to avoid circular import
if TYPE_CHECKING:
    from schub.model.base import BaseModelClass


class TrainRunner:
    """TrainRunner"""

    _train_cls = Trainer

    def __init__(
        self,
        model: "BaseModelClass",
        data_splitter: DataSplitter,
        training_arguments: TrainingArguments,
        data_collator: Optional[Callable] = default_data_collator,
        **trainer_kwargs,
    ):
        self.model = model
        self.data_splitter = data_splitter
        self.training_arguments = training_arguments

        # split the ann dataset
        train_dataset, valid_dataset, _ = data_splitter.split()

        if not hasattr(self.model, "module"):
            raise AttributeError("The module is not initialized in your model.")
        self.trainer = self._train_cls(
            self.model.module,
            args=training_arguments,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            **trainer_kwargs,
        )

    def __call__(self, *args, **kwargs):
        """Caller of model training"""
        self.model.module.train()
        self.trainer.train(*args, **kwargs)

        # data splitter gets these attributes after training
        self.model.train_indices = self.data_splitter.train_idx
        self.model.validation_indices = self.data_splitter.val_idx
        self.model.test_indices = self.data_splitter.test_idx

        self.model.module.eval()
        self.model.is_trained_ = True
        self.model.trainer = self.trainer
