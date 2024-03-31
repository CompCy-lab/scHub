from __future__ import annotations

import logging

import numpy as np
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

from schub import REGISTRY_KEYS
from schub.data import AnnDataManager, SampleAnnData, SampleAnnTorchDataset
from schub.data.fields import CategoricalSampleField, LayerField
from schub.dataloader import RandomDataSplitter
from schub.module import DeepSet, DeepSetConfig
from schub.train import TrainRunner

from .base import BaseModelClass, ModuleMixin

logger = logging.getLogger(__name__)


class CytoSet(ModuleMixin, BaseModelClass):
    """CytoSet Model (https://dl.acm.org/doi/abs/10.1145/3459930.3469529)

    CytoSet is deep learning model used for predicting clinical outcomes from set-based single-cell data.
    """

    module_cls = DeepSet

    def __init__(
        self,
        adata: SampleAnnData | None = None,
        config: DeepSetConfig | None = None,
    ):
        super().__init__(adata, config)

        if config is not None:
            self._model_summary_string = "CytoSet model with the following params: \n" + config.to_json_string()

    def _init_module(self, config: PretrainedConfig) -> PreTrainedModel:
        """Init a deepset module from config"""
        module = self.module_cls(config)

        return module

    @classmethod
    def setup_anndata(
        cls,
        adata: SampleAnnData,
        label_key: str,
        layer: str | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalSampleField(REGISTRY_KEYS.LABELS_KEY, label_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        num_epochs: int,
        per_device_batch_size: int,
        learning_rate: float = 1e-3,
        train_size: float = 0.6,
        validation_size: float | None = None,
        random_state: int = 42,
        save_dir: str = "./training_logs",
        resume_from_checkpoint: bool = False,
        **training_args,
    ):
        logger.info(f"Training for {num_epochs} epochs.")

        ann_dataset = SampleAnnTorchDataset(
            adata_manager=self.adata_manager,
            set_size=self.config.set_size,
            getitem_tensors={
                REGISTRY_KEYS.X_KEY: np.float32,
                REGISTRY_KEYS.LABELS_KEY: np.int64,
            },
            random_state=random_state,
        )
        data_splitter = RandomDataSplitter(
            ann_dataset, train_size=train_size, validation_size=validation_size, random_state=random_state
        )

        training_arguments = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=num_epochs,
            per_gpu_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            label_names=[REGISTRY_KEYS.LABELS_KEY],
            seed=random_state,
            evaluation_strategy="steps",
            **training_args,
        )

        # def compute_metrics(p):
        #     print(p.predictions)
        #     print(p.label_ids)
        #
        #     return {
        #         "eval_f1": 0.8,
        #         "eval_precision": 0.9,
        #     }

        trainer = TrainRunner(
            self,
            training_arguments=training_arguments,
            data_splitter=data_splitter,
            # compute_metrics=compute_metrics,
        )

        return trainer(resume_from_checkpoint=resume_from_checkpoint, ignore_keys_for_eval=["hidden_state"])
