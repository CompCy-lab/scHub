from typing import Optional

from anndata import AnnData
from scanpy._settings import settings


def _get_rep_columns(
    adata: AnnData,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
) -> list[str]:
    """Get the column names"""
    column_names = adata.var_names.tolist()
    if use_rep is None and n_pcs == 0:
        return column_names
    if use_rep is None:
        if adata.n_vars > settings.N_PCS:
            n_columns = n_pcs
            if "X_pca" in adata.obsm.keys():
                n_columns = min(n_columns, adata.obsm["X_pca"].shape[1])
            elif n_pcs is None:
                n_columns = settings.N_PCS
            column_names = [f"X_pca{i}" for i in range(n_columns)]
        else:
            return column_names
    else:
        if use_rep in adata.obsm.keys() and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f"{use_rep} does not have enough Dimensions. Provide a "
                    "Representation with equal or more dimensions than"
                    "`n_pcs` or lower `n_pcs` "
                )
            column_names = [f"X_pca{i}" for i in range(n_pcs)]
        elif use_rep in adata.obsm.keys() and n_pcs is None:
            column_names = [f"{use_rep}_{i}" for i in range(adata.obsm[use_rep].shape[1])]
        elif use_rep == "X":
            return column_names
        else:
            raise ValueError(f"Did not find {use_rep} in `.obsm.keys()`. " "You need to compute it first.")

    return column_names
