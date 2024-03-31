import inspect
import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional
from uuid import uuid4

import numpy as np
import rich
from anndata import AnnData
from transformers import PretrainedConfig, PreTrainedModel

from schub.data import AnnDataManager
from schub.data._constants import (
    _MODEL_NAME_KEY,
    _SCHUB_UUID_KEY,
    _SETUP_ARGS_KEY,
    _SETUP_METHOD_NAME,
)
from schub.data._types import AnnOrMuData
from schub.data._utils import _assign_adata_uuid, _check_if_view

logger = logging.getLogger(__name__)

_SETUP_INPUTS_EXCLUDED_PARAMS = {"adata", "mdata", "kwargs"}


class BaseModelMetaClass(ABCMeta):
    """Metaclass for the Abstract Base Class"""

    @abstractmethod
    def __init__(cls, name, bases, dct):
        cls._setup_adata_manager_store: dict[
            str, type[AnnDataManager]
        ] = {}  # Maps the adata id to AnnDataManager instances
        cls._per_instance_manager_store: dict[
            str, dict[str, type[AnnDataManager]]
        ] = {}  # Maps model instance id to AnnDataManager mappings.
        super().__init__(name, bases, dct)


class BaseModelClass(metaclass=BaseModelMetaClass):
    """Abstract class for the Models"""

    def __init__(
        self,
        adata: Optional[AnnOrMuData] = None,
        config: Optional[PretrainedConfig] = None,
    ):
        self.id = str(uuid4())
        if adata is not None:
            self._adata = adata
            self._adata_manager = self._get_most_recent_adata_manager(adata, required=True)
            self._register_manager_for_instance(self.adata_manager)
            self.registry_ = self._adata_manager.registry
            self.summary_stats = self._adata_manager.summary_stats

        self.is_trained_ = False
        self._model_summary_string = "Model is not configured" if config is None else ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        # self._data_set_cls = AnnTorchDataset
        # self._data_loader_cls = AnnDataLoader

        self._config = config
        if config is not None:
            try:
                self.module = self._init_module(config)
            except RuntimeError as err:
                raise RuntimeError("cannot initialize module from the input config.") from err

    @abstractmethod
    def _init_module(self, config: PretrainedConfig) -> PreTrainedModel:
        """Init module used in the model including parameter initialization"""
        pass

    def _init_module_from_pretrained(
        self,
        *args,
        module_cls=None,
        **kwargs,
    ):
        if module_cls is None:
            if not hasattr(self, "module_cls"):
                raise ValueError("To initialize module, ``module_cls`` cannot be None.")
            module_cls = self.module_cls
        # for arg in args:
        #     print(arg)
        # for key in kwargs:
        #     print(key, kwargs[key])

        self.module = module_cls.from_pretrained(*args, **kwargs)

        if hasattr(self.module, "config"):
            self._config = self.module.config
            self._model_summary_string = (
                f"{self.__class__.__name__} model with the following params: \n" + self.config.to_json_string()
            )
        else:
            warnings.warn(
                "The model configuration is neither provided " "from the pre-trained model nor from the input args.",
                UserWarning,
                stacklevel=2,
            )
        # set the training status to true
        self.is_trained_ = True

    @property
    def config(self) -> PretrainedConfig:
        """Get the config (PretrainedConfig) used to initialize the PretrainedModel"""
        return self._config

    @config.setter
    def config(self, config: PretrainedConfig):
        if config is None:
            raise ValueError("config cannot be None.")
        if self.config is not None:
            logger.info("initialize the model from the new config.")
        self._init_module(config)
        self._config = config

    def view_model_config(self):
        rich.print(self.config.to_json_string())
        return ""

    @property
    def adata(self) -> AnnOrMuData:
        """Get the data attached to the model instance"""
        return self._adata

    @adata.setter
    def adata(self, adata: AnnOrMuData):
        if adata is None:
            raise ValueError("adata cannot be None.")
        self._validate_anndata(adata)
        self._adata = adata
        self._adata_manager = self.get_anndata_manager(adata, required=True)
        self.registry_ = self._adata_manager.registry
        self.summary_stats = self._adata_manager.summary_stats

    @property
    def adata_manager(self) -> AnnDataManager:
        """Get the manager instance associated with `self.data`"""
        return self._adata_manager

    @staticmethod
    def _get_setup_method_args(**setup_locals) -> dict:
        """Returns a dict that organizes the arguments used to call `setup_anndata`."""
        cls = setup_locals.pop("cls")
        method_name = None
        if "adata" in setup_locals:
            method_name = "setup_anndata"
        elif "mdata" in setup_locals:
            method_name = "setup_mudata"

        model_name = cls.__name__
        setup_args = {}
        for k, v in setup_locals.items():
            if k not in _SETUP_INPUTS_EXCLUDED_PARAMS:
                setup_args[k] = v
        return {
            _MODEL_NAME_KEY: model_name,
            _SETUP_METHOD_NAME: method_name,
            _SETUP_ARGS_KEY: setup_args,
        }

    @classmethod
    def register_manager(cls, adata_manager: AnnDataManager):
        """Registers an AnnDataManager instance with this model class"""
        adata_id = adata_manager.adata_uuid
        cls._setup_adata_manager_store[adata_id] = adata_manager

    def _register_manager_for_instance(self, adata_manager: AnnDataManager):
        """Register an AnnData Manager associated with the instance id"""
        if self.id not in self._per_instance_manager_store:
            self._per_instance_manager_store[self.id] = {}

        adata_id = adata_manager.adata_uuid
        instance_manager_store = self._per_instance_manager_store[self.id]
        instance_manager_store[adata_id] = adata_manager

    def deregister_manager(self, adata: Optional[AnnData] = None):
        """Deregister the `AnnDataManager` instance associated with `adata`"""
        cls_manager_store = self._setup_adata_manager_store
        instance_manager_store = self._per_instance_manager_store

        if adata is None:
            instance_managers_to_clear = list(instance_manager_store.keys())
            cls_managers_to_clear = list(cls_manager_store.keys())
        else:
            adata_manager = self._get_most_recent_adata_manager(adata, required=True)
            cls_managers_to_clear = [adata_manager.adata_uuid]
            instance_managers_to_clear = [adata_manager.adata_uuid]

        for adata_id in cls_managers_to_clear:
            # don't clear the current manager by default
            is_current_adata = adata is None and adata_id == self.adata_manager.adata_uuid
            if is_current_adata or adata_id not in cls_manager_store:
                continue
            del cls_manager_store[adata_id]

        for adata_id in instance_managers_to_clear:
            # don't clear the current manager by default
            is_current_adata = adata is None and adata_id == self.adata_manager.adata_uuid
            if is_current_adata or adata_id not in instance_manager_store:
                continue
            del instance_manager_store[adata_id]

    @classmethod
    def _get_most_recent_adata_manager(
        cls,
        adata: AnnOrMuData,
        required: bool = False,
    ) -> Optional[AnnDataManager]:
        """Retrieves the anndata manager for a given AnnData object specific to this model class."""
        if _SCHUB_UUID_KEY not in adata.uns:
            if required:
                raise ValueError(f"Please set up your AnnData with {cls.__name__}.setup_anndata first.")
            return None

        adata_id = adata.uns[_SCHUB_UUID_KEY]

        if adata_id not in cls._setup_adata_manager_store:
            if required:
                raise ValueError(
                    f"Please set up your AnnData with {cls.__name__}.setup_anndata first. "
                    "It appears that the AnnData object has been setup with a different model."
                )
            return None

        adata_manager = cls._setup_adata_manager_store[adata_id]
        if adata_manager.adata is not adata:
            raise ValueError(
                "The supplied AnnData object does not match the AnnData object "
                "supplied during setup. Have you created a copy?"
            )

        return adata_manager

    def get_anndata_manager(
        self,
        adata: AnnOrMuData,
        required: bool = False,
    ) -> Optional[AnnDataManager]:
        cls = self.__class__
        if _SCHUB_UUID_KEY not in adata.uns:
            if required:
                raise ValueError(f"Please set up your AnnData with {cls.__name__}.setup_anndata first.")
            return None

        adata_id = adata.uns[_SCHUB_UUID_KEY]
        if self.id not in cls._per_instance_manager_store:
            if required:
                raise AssertionError(
                    "Unable to find instance specific manager store. "
                    "The model has likely not been initialized with an AnnData object."
                )
            return None
        elif adata_id not in cls._per_instance_manager_store[self.id]:
            if required:
                raise AssertionError("Please call ``self._validate_anndata`` on this AnnData object.")
            return None

        adata_manager = cls._per_instance_manager_store[self.id][adata_id]
        if adata_manager.adata is not adata:
            logger.info("AnnData object appears to be a copy. Attempting to transfer setup.")
            _assign_adata_uuid(adata, overwrite=True)
            adata_manager = self.adata_manager.transfer_fields(adata)
            self._register_manager_for_instance(adata_manager)

        return adata_manager

    def get_from_registry(
        self,
        adata: AnnOrMuData,
        registry_key: str,
    ) -> np.ndarray:
        """
        Returns the object in AnnData associated with the key in the data registry

        AnnData object should be registered with model prior to calling this function
        via the ``self._validate_anndata`` method.

        Parameter
        ---------
        registry_key
            key of object to get from data registry.
        adata
            AnnData to pull data from.

        Returns
        -------
        The requested data as a Numpy array.
        """
        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            raise AssertionError(
                "AnnData not registered with model. Call `self._validate_anndata` " "prior to calling this function."
            )

        return adata_manager.get_from_registry(registry_key)

    def _make_data_loader(
        self,
        adata: AnnOrMuData,
    ):
        pass

    def _validate_anndata(
        self,
        adata: Optional[AnnOrMuData] = None,
        copy_if_view: bool = True,
    ) -> AnnOrMuData:
        """Validate anndata has been properly registered, transfer if necessary"""
        if adata is None:
            adata = self.adata

        _check_if_view(adata, copy_if_view=copy_if_view)

        adata_manager = self.get_anndata_manager(adata)
        if adata_manager is None:
            logger.info("Input AnnData is not set up. " "attempting to transfer AnnData setup")
            self._register_manager_for_instance(self.adata_manager.transfer_fields(adata))
        else:
            adata_manager.validate()

        return adata

    def check_if_trained(
        self,
        warn: bool = True,
        message: str = "model is not trained, please train the model first.",
    ):
        """Check if the model is trained."""
        if not self.is_trained:
            if warn:
                warnings.warn(message, UserWarning, stacklevel=2)
            else:
                raise RuntimeError(message)

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @is_trained.setter
    def is_trained(self, value):
        self.is_trained_ = value

    @property
    def train_indices(self) -> np.ndarray:
        """Get indices in the training set"""
        return self.train_indices_

    @train_indices.setter
    def train_indices(self, value):
        self.train_indices_ = value

    @property
    def test_indices(self) -> np.ndarray:
        """Get indices in the testing set"""
        return self.test_indices_

    @test_indices.setter
    def test_indices(self, value):
        self.test_indices_ = value

    @property
    def validation_indices(self) -> np.ndarray:
        """Get indices in the validation set"""
        return self.validation_indices_

    @validation_indices.setter
    def validation_indices(self, value):
        self.validation_indices_ = value

    @property
    def history(self):
        """Returns computed metrics during training."""
        return self.history_

    def _get_user_attributes(self):
        """Returns all the self attributes defined in a model class, e.g., `self.is_trained_`."""
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    @abstractmethod
    def train(self):
        """Trains the model."""

    @property
    def summary_string(self):
        """Summary string of the model."""
        summary_string = self._model_summary_string
        summary_string += "\nTraining status: {}".format("Trained" if self.is_trained_ else "Not Trained")
        return summary_string

    def __repr__(self):
        rich.print(self.summary_string)
        return ""

    @classmethod
    @abstractmethod
    def setup_anndata(
        cls,
        adata: AnnOrMuData,
        *args,
        **kwargs,
    ):
        """Class and Abstract method for setting up the data used for training

        Every model deriving from this class provides parameters to this method with
        the parameters it requires. For correct model initialization, the implementation
        must call `cls.register_manager` on a model-specific instance of `AnnDataManager`
        """
        pass

    def view_anndata_setup(self, adata: Optional[AnnOrMuData] = None, hide_state_registries: bool = False) -> None:
        """
        Print summary of the setup for the initial AnnData or a given AnnData object.

        Parameters
        ----------
        adata
            AnnData object setup with ``setup_anndata``.
        hidden_state_registries
            If True, prints a shortened summary without details of each state registry
        """
        if adata is None:
            adata = self.adata
        try:
            adata_manager = self.get_anndata_manager(adata, required=True)
        except ValueError as err:
            raise ValueError(
                f"Given AnnData not setup with {self.__class__.__name__}. " "Cannot view setup summary."
            ) from err
        adata_manager.view_registry(hide_state_registries=hide_state_registries)
