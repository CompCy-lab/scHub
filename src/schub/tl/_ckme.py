from typing import Optional

import pandas as pd
from anndata import AnnData
from scanpy._utils import AnyRandom
from scanpy.tools._utils import _choose_representation
from scipy.sparse import issparse

from schub.transform import RandomFourierTransform


def ckme(
    adata: AnnData,
    partition_key: str,
    gamma: float = 1,
    rff_dim: int = 1000,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    random_state: AnyRandom = 0,
    key_added: Optional[str] = None,
    copy=False,
):
    r"""Cell Kernel Mean Embedding (CKME) :cite:`shan2022transparent`.

    CKME uses random fourier transformation :cite:`rahimi2007random` to transform the feature matrix of each sample.
    The kernel CKME uses is RBF kernel, which is defined as:

    .. math::

       K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma^2 \|\mathbf{x} - \mathbf{y}\|^2)

    Parameters
    ----------
    adata
        Annotated data of type `anndata.AnnData`
    partition_key
        Column key in the field of `adata.obs` indicating the sample ids
    gamma
        Gamma parameter for the RBF kernel
    rff_dim
        Dimensionality of the random Fourier features
    n_pcs
        Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically:
        For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
        If 'X_pca' is not present, it's computed with default parameters.
    random_state
        A random seed which supports an `int`, `None` and `numpy.random.RandomState`
    key_added
        If not specified, the results are stored in `.uns['ckme']`.
        If specified, the results are stored in `.uns[key_added+'_ckme']`.
    copy
        Return a copy instead of writing the results into `adata`.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following fields.

    **X_ckme**: `adata.obsm` field
        Resulting random Fourier feature for each cell
    **ckme** or **{key_added}_ckme**: `adata.uns` field
        results from cell kernel mean embedding, including both parameters and sample embeddings
    """
    adata = adata.copy() if copy else adata

    if key_added is None:
        key_added = "ckme"
    else:
        key_added = key_added + "_ckme"

    adata.uns[key_added] = {}
    ckme_dict = adata.uns[key_added]

    ckme_dict["partition_key"] = partition_key
    ckme_dict["params"] = {"gamma": gamma, "rff_dim": rff_dim}
    ckme_dict["random_state"] = random_state

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, silent=True)
    if issparse(X):
        X = X.toarray()
    if use_rep is not None:
        ckme_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        ckme_dict["params"]["n_pcs"] = n_pcs

    X_ckme = RandomFourierTransform(gamma=gamma, rff_dim=rff_dim, random_state=random_state).fit_transform(X)

    adata.obsm["X_ckme"] = X_ckme

    # save the rff features based on partition_key (sample_id)
    df_partition = pd.DataFrame(
        adata.obsm["X_ckme"].copy(),
        columns=[f"X_ckme_{i}" for i in range(adata.obsm["X_ckme"].shape[1])],
    )
    df_partition[partition_key] = adata.obs[partition_key].tolist()
    df_partition = df_partition.groupby(partition_key, sort=False).mean()
    ckme_dict[f"{partition_key}_ckme"] = df_partition

    return adata if copy else None
