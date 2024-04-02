import os
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import read_h5ad

import schub
from schub.data import SampleAnnData
from schub.model import CytoSet
from schub.module import DeepSetConfig

HERE: Path = Path(__file__).parent
DATA_ROOT = os.path.join(HERE, "_data")


class TestDeepLearningModels:
    def test_cytoset(self):
        # read and set up the data first
        adata = read_h5ad(os.path.join(DATA_ROOT, "adata-pbmc.h5ad"))
        sadata = SampleAnnData(
            adata,
            sampleid="str_labels",
        )
        sadata.obs_to_sample(columns="labels")
        # preprocess the data
        sc.pp.normalize_total(sadata, target_sum=10000)
        sc.pp.log1p(sadata)

        CytoSet.setup_anndata(sadata, label_key="labels")
        adata_manager = CytoSet._get_most_recent_adata_manager(sadata, required=True)
        # adata_manager.view_registry()
        label_registry = adata_manager.get_state_registry(registry_key="label")
        num_labels = len(label_registry.categorical_mapping)

        # create the model using the config
        config = DeepSetConfig(
            input_size=sadata.n_vars,
            set_size=64,
            hidden_size=128,
            n_label=num_labels,
        )
        # test training loop
        save_dir = os.path.join(HERE, "pretrained_models", "cytoset")
        model = CytoSet(config=config, adata=sadata)
        model.train(num_epochs=16, per_device_batch_size=3, save_dir=save_dir, logging_steps=10, save_steps=10)
        # model.trainer.evaluate()
        model.save_pretrained(
            save_directory=os.path.join(save_dir, "pretrained_model"), save_anndata=True, overwrite=True
        )

        # test resuming the training from checkpoint
        model = CytoSet.from_pretrained(
            os.path.join(save_dir, "checkpoint-20"), adata=os.path.join(save_dir, "pretrained_model", "adata.h5ad")
        )
        model.train(
            num_epochs=24,
            per_device_batch_size=3,
            save_dir=save_dir,
            logging_steps=1,
            resume_from_checkpoint=os.path.join(save_dir, "checkpoint-20"),
        )


class TestMachineLearningModels:
    def test_ckme(self):
        adata = sc.datasets.pbmc3k_processed()

        rff_dim, gamma = 128, 1.0
        schub.tl.ckme(adata, partition_key="louvain", gamma=gamma, rff_dim=rff_dim, use_rep="X_pca")
        assert "ckme" in adata.uns.keys()
        assert "X_ckme" in adata.obsm.keys()
        assert adata.obsm["X_ckme"].shape == (adata.n_obs, rff_dim)
        output_pca = adata.obsm["X_ckme"].copy()

        schub.tl.ckme(adata, partition_key="louvain", gamma=gamma, rff_dim=rff_dim, use_rep="X")
        output_X = adata.obsm["X_ckme"].copy()
        assert not np.array_equal(output_pca, output_X)

        new_rff_dim = 64
        schub.tl.ckme(adata, partition_key="louvain", gamma=gamma, rff_dim=new_rff_dim)
        assert adata.obsm["X_ckme"].shape == (adata.n_obs, new_rff_dim)

    def test_cytoemd(self):
        adata = read_h5ad(os.path.join(DATA_ROOT, "adata-bcrxl.h5ad"))

        # add spawn to avoid deadlock
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        # run cytoemd method
        n_components = 2
        num_samples = len(adata.obs["fcs_files"].unique())
        schub.tl.cytoemd(
            adata,
            partition_key="fcs_files",
            embedding_method="umap",
            n_components=n_components,
            random_state=0,
            n_jobs=4,
        )
        # test if results exist
        assert "cytoemd" in adata.uns.keys()
        # check result size
        assert adata.uns["cytoemd"]["X_emd"].shape == (num_samples, n_components)
        assert adata.uns["cytoemd"]["emd"].shape == (num_samples, num_samples)

    def test_sketch(self):
        adata = sc.datasets.pbmc3k_processed()

        # geometric sketching
        schub.pp.sketch(adata, n_sketch=256, use_rep="X_pca", method="gs", key_added="gs")
        adata_sketch_gs = adata[adata.obs["gs_sketch"]]
        assert adata_sketch_gs.n_obs == 256

        # kernel herding
        schub.pp.sketch(adata, n_sketch=128, use_rep="X_pca", method="kernel_herding", key_added="kh")
        adata_sketch_kh = adata[adata.obs["kh_sketch"]]
        assert adata_sketch_kh.n_obs == 128

        # random sampling
        schub.pp.sketch(adata, n_sketch=512, use_rep="X_pca", method="random", key_added="random")
        adata_sketch_random = adata[adata.obs["random_sketch"]]
        assert adata_sketch_random.n_obs == 512

    def test_sclkme(self):
        adata = sc.datasets.pbmc3k_processed()

        sketch_size = 128
        schub.pp.sketch(adata, n_sketch=sketch_size, use_rep="X_pca")
        X_anchor = adata[adata.obs["sketch"]].obsm["X_pca"].copy()
        schub.tl.kernel_mean_embedding(adata, partition_key="louvain", use_rep="X_pca", X_anchor=X_anchor)

        num_sample = len(adata.obs["louvain"].unique())
        assert adata.uns["kme"]["louvain_kme"].shape == (num_sample, sketch_size)

    def test_delve(self):
        adata = read_h5ad(os.path.join(DATA_ROOT, "adata_RPE.h5ad"))
        num_subsamples = 1000

        # avoid deadlock by set spawn
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)

        schub.pp.delve(
            adata, knn=10, num_subsamples=num_subsamples, use_rep="X", n_clusters=5, random_state=0, n_jobs=-1
        )

        assert adata.uns["delve"]["delta_mean"].shape == (num_subsamples, adata.n_vars)
        assert "delve" in adata.var.keys()
