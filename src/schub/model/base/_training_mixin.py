from __future__ import annotations

import logging

from transformers import TrainingArguments

from schub.data import AnnTorchDataset
from schub.dataloader import RandomDataSplitter
from schub.train import TrainRunner

logger = logging.getLogger(__name__)


class ModelTrainingMixin:
    """A general purpose train method for neural network models"""

    def train(
        self,
        output_dir: str,
        per_device_batch_size: int,
        num_train_epochs: float = 3.0,
        *,
        max_steps: int = -1,
        train_size: float = 0.8,
        validation_size: float | None = None,
        learning_rate: float | None = 1e-3,
        seed: int | None = 32,
        **training_args,
    ):
        logger.info(f"Training for {num_train_epochs} epochs.")

        # initialize torch dataset and dataloader
        ann_dataset = AnnTorchDataset(self.adata_manager)
        # split the data into train and validation
        data_splitter = RandomDataSplitter(
            ann_dataset,
            train_size=train_size,
            validation_size=validation_size,
        )

        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            seed=seed,
            **training_args,
        )

        trainer = TrainRunner(
            self,
            data_splitter=data_splitter,
            training_arguments=training_arguments,
        )

        return trainer()
