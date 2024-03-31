import os
from pathlib import Path
from typing import Optional, Union

import torch
from anndata import AnnData, read_h5ad
from mudata import MuData
from transformers import PretrainedConfig

from schub.data import SampleAnnData
from schub.data._constants import (
    _MODEL_NAME_KEY,
    _SETUP_ARGS_KEY,
    _SETUP_METHOD_NAME,
    UNS_SAMPLE_KEY,
)
from schub.data._types import AnnOrMuData


class ModuleMixin:
    def to_device(self, device: Union[str, int]):
        """Move model to device."""
        my_device = torch.device(device)
        self.module.to(my_device)

    @property
    def device(self) -> torch.device:
        return self.module.device

    def save_pretrained(
        self,
        save_directory: Union[os.PathLike, str],
        data_prefix: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        push_to_hub: bool = False,
        **save_pretrained_kwargs,
    ):
        if not os.path.exists(save_directory) or overwrite:
            os.makedirs(save_directory, exist_ok=overwrite)
        else:
            raise FileExistsError(
                f"{str(save_directory)} already exists, please provide an not existed dir for saving model."
            )

        data_name_prefix = data_prefix or ""
        if save_anndata:
            file_suffix = ""
            if isinstance(self.adata, AnnData):
                file_suffix = "_adata.h5ad" if data_prefix is not None else "adata.h5ad"
            if isinstance(self.adata, MuData):
                file_suffix = "_mdata.h5mu" if data_prefix is not None else "mdata.h5mu"
            self.adata.write_h5ad(
                filename=os.path.join(save_directory, f"{data_name_prefix}{file_suffix}"),
                compression="gzip",
            )

        self.module.save_pretrained(save_directory, push_to_hub=push_to_hub, **save_pretrained_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        adata: Union[AnnOrMuData, str, Path] = None,
        *module_args,
        module_cls=None,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        **from_pretrained_kwargs,
    ):
        # load the data
        if adata is None:
            adata = os.path.join(pretrained_model_name_or_path, "adata.h5ad")

        if isinstance(adata, str) or isinstance(adata, Path):
            new_adata = read_h5ad(adata)
            if UNS_SAMPLE_KEY in new_adata.uns:
                sampleid = new_adata.uns[UNS_SAMPLE_KEY].index.name
                new_adata = SampleAnnData(
                    new_adata,
                    sampleid=sampleid,
                    samplem=new_adata.uns[UNS_SAMPLE_KEY],
                )
        else:
            new_adata = adata

        if _MODEL_NAME_KEY in new_adata.uns and new_adata.uns[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in new_adata.uns:
            raise ValueError("Saved model doesn't contain original setup inputs. " "Cannot load the original setup.")
        # Calling ``setup_anndata`` method with the original arguments passed into
        # the saved model. This enables simple backwards compatibility in the case of
        # newly introduced fields or parameters.
        method_name = new_adata.uns.get(_SETUP_METHOD_NAME, "setup_anndata")
        setup_kwargs = new_adata.uns[_SETUP_ARGS_KEY]
        getattr(cls, method_name)(
            new_adata,
            **setup_kwargs,
        )

        # initialize the model
        model = cls(adata=new_adata, config=None)
        model._init_module_from_pretrained(
            pretrained_model_name_or_path,
            *module_args,
            module_cls=module_cls,
            config=config,
            **from_pretrained_kwargs,
        )

        return model
