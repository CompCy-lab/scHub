import logging
import multiprocessing as mp
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from scanpy._utils import AnyRandom
from scanpy.tools._utils import _choose_representation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from tqdm import tqdm

from schub.utils import _get_rep_columns

logger = logging.getLogger(__name__)


def delve(
    adata: AnnData,
    knn: int = 10,
    num_subsamples: int = 1000,
    n_clusters: int = 5,
    null_iterations: int = 1000,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    n_kmeans_init: int = 10,
    random_state: AnyRandom = 42,
    n_jobs: int = -1,
    key_added: Optional[str] = None,
    copy: bool = False,
):
    """DELVE feature selection methods :cite:`ranek2023feature`.

    Specifically, DELVE includes the following two steps:
        - step 1: identifies dynamic seed features to construct a between-cell affinity graph according to
                  dynamic cell state progression
        - step 2: ranks features according to their total variation in signal along the approximate trajectory graph
                  using the Laplacian score

    Parameters
    ----------
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data (dimensions = cells x features)
    knn: int
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int
        number of neighborhoods to subsample when estimating feature dynamics
    n_clusters: int
        number of feature modules
    null_iterations: int
        number of iterations for gene-wise permutation testing
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically:
        For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
        If 'X_pca' is not present, it's computed with default parameters.
    random_state: `AnyRandom`
        random seed or state
    n_kmeans_init: int
        number of kmeans clustering initializations
    n_pcs: int
        number of principal components to compute pairwise Euclidean distances for between-cell
        affinity graph construction. If None, uses `adata.X`
    n_jobs: int
        number of tasks
    key_added
        If not specified, the parameters and `delta_mean` are stored in `.uns['delve']`.
        If specified, the parameters and `delta_mean` are stored in `.uns[key_added+'delve']`,
        and (feature `modules`, ranked scores) are stored in (`.var['delve_cluster_id']`
        and `.var['delve_cluster_permutation_pval']`) or (`.var[key_added+'_delve_cluster_id'],
        `.var[key_added+"_delve_cluster_permutation_pval"]`), respectively.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    delta_mean: `pd.DataFrame`
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods
        (dimensions = num_subsamples x features)
    modules: `pd.DataFrame`
        dataframe containing feature-cluster assignments and permutation p-values (dimensions = features x 2)
    selected_features: `pd.DataFrame`
        dataframe containing ranked features and Laplacian scores
        following feature selection (dimensions = features x 1)

    """
    adata = adata.copy() if copy else adata
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if key_added is None:
        key_added = "delve"
    else:
        key_added = key_added + "_delve"

    adata.uns[key_added] = {}
    delve_dict = adata.uns[key_added]

    delve_dict.update(
        {
            "params": {
                "knn": knn,
                "num_subsamples": num_subsamples,
                "n_clusters": n_clusters,
                "null_iterations": null_iterations,
                "n_kmeans_init": n_kmeans_init,
            }
        }
    )
    if use_rep is not None:
        adata.uns[key_added]["use_rep"] = use_rep
    if n_pcs is not None:
        adata.uns[key_added]["n_pcs"] = n_pcs

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, silent=True)
    obs_names = adata.obs_names.to_numpy()
    feature_names = np.array(_get_rep_columns(adata, use_rep=use_rep, n_pcs=n_pcs))
    # check the random state and get the seed
    random_state = check_random_state(random_state)
    random_seed = random_state.get_state()[1][0]
    adata.uns[key_added]["random_state"] = random_seed

    # identify dynamic feature modules
    sub_idx, _, delta_mean, modules = seed_select(
        X,
        feature_names=feature_names,
        obs_names=obs_names,
        k=knn,
        num_subsamples=num_subsamples,
        n_clusters=n_clusters,
        null_iterations=null_iterations,
        random_state=random_seed,
        n_random_state=n_kmeans_init,
        n_jobs=n_jobs,
    )

    dyn_feats = np.asarray(modules.index[modules["cluster_id"] != "static"])
    selected_features = feature_select(
        X=X[sub_idx, :], feature_names=feature_names, dyn_feats=dyn_feats, k=knn, n_jobs=n_jobs
    )

    # save the results: modules and ranked features to adata.var
    delve_dict["delta_mean"] = delta_mean
    selected_features.rename({"DELVE": key_added}, axis=1, inplace=True)
    adata.var.loc[selected_features.index, key_added] = selected_features[key_added]

    # module_columns = [f"{key_added}_{col}" for col in modules.columns]
    # modules.columns = module_columns
    mapper = {col: f"{key_added}_{col}" for col in modules.columns}
    modules.rename(mapper, axis=1, inplace=True)
    for col in modules.columns:
        adata.var.loc[modules.index, col] = modules[col]

    return adata if copy else None


def feature_select(
    X: np.ndarray = None,
    feature_names: np.ndarray = None,
    dyn_feats: np.ndarray = None,
    k: int = 10,
    n_pcs: Optional[int] = None,
    n_jobs: int = -1,
):
    """Ranks features along dynamic seed graph using the Laplacian score :cite:`he2005laplacian`.

    Parameters
    ----------
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    dyn_feats: np.ndarray (default = None)
        array containing features that are dynamically expressed.
        Can consider replacing this with a set of known regulators.
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity
        graph construction. If None, uses adata.X
    n_jobs: int (default = -1)
        number of tasks

    Returns
    -------
    selected_features: `pd.DataFrame`
        dataframe containing ranked features and Laplacian scores for feature selection (dimensions = features x 1)

    """
    f_idx = np.where(np.isin(feature_names, dyn_feats))[0]  # index of feature names to construct seed graph
    # constructs graph using dynamic seed features
    W = construct_affinity(X=X[:, f_idx], k=k, n_pcs=n_pcs, n_jobs=n_jobs)
    scores = laplacian_score(X=X, W=W)
    selected_features = pd.DataFrame(scores, index=feature_names, columns=["DELVE"])
    selected_features = selected_features.sort_values(by="DELVE", ascending=True)

    return selected_features


def seed_select(
    X: np.ndarray,
    feature_names: Optional[np.ndarray] = None,
    obs_names: Optional[np.ndarray] = None,
    k: int = 10,
    num_subsamples: int = 1000,
    n_clusters: int = 5,
    null_iterations: int = 1000,
    random_state: int = 0,
    n_random_state: int = 10,
    n_pcs: int = None,
    n_jobs: int = -1,
):
    """Identifies dynamic seed clusters

    Parameters
    ----------
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    random_state: int (default = 0)
        random seed parameter
    n_random_state: int (default = 10)
        number of kmeans clustering initializations
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph
        construction. If None, uses adata.X
    n_jobs: int (default = -1)
        number of tasks

    Returns
    -------
    sub_idx: np.ndarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: `pd.DataFrame`
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods
        (dimensions = num_subsamples x features)
    modules: `pd.DataFrame`
        dataframe containing feature-cluster assignments and permutation p-values (dimensions = features x 2)

    """
    if n_jobs == -1:
        n_jobs = max(mp.cpu_count() // 2, 1)  # heuristically set jobs as half of total cpu cores

    np.random.seed(random_state)
    random_state_arr = np.random.randint(0, 1000000, n_random_state)

    logging.info("estimating feature dynamics")
    sub_idx, adata_sub, delta_mean = delta_exp(
        X=X,
        feature_names=feature_names,
        obs_names=obs_names,
        k=k,
        num_subsamples=num_subsamples,
        random_state=random_state,
        n_pcs=n_pcs,
        n_jobs=n_jobs,
    )

    # identify modules
    mapping_df = pd.DataFrame(index=feature_names)
    pval_df = pd.DataFrame(index=feature_names)
    dyn_feats, random_state_idx = [], []

    p = mp.Pool(n_jobs)
    results = p.imap(partial(_run_cluster, delta_mean, feature_names, n_clusters, null_iterations), random_state_arr)

    desc = "clustering features and performing feature-wise permutation testing"
    with tqdm(range(n_random_state), desc=desc) as t:
        for result in results:
            if result is not None:
                mapping_df = pd.concat([mapping_df, result[0]], axis=1)
                pval_df = pd.concat([pval_df, result[1]], axis=1)
                dyn_feats.append(result[2])
                random_state_idx.append(result[3])
            t.update()

    p.close()
    p.join()

    if len(dyn_feats) == 0:
        warn_msg = """No feature clusters have a dynamic variance greater than null.
            Consider changing the number of clusters or the subsampling size."""
        warnings.warn(warn_msg, UserWarning, stacklevel=2)
    else:
        dyn_feats = list(np.unique(list(set.intersection(*map(set, dyn_feats)))))
        if len(dyn_feats) == 0:
            warnings.warn(
                "No features were considered dynamically-expressed across runs.", category=UserWarning, stacklevel=2
            )
        else:
            modules = _annotate_clusters(
                mapping_df=mapping_df, dyn_feats=dyn_feats, pval_df=pval_df, random_state_idx=random_state_idx[-1]
            )
            n_dynamic_clusters = len(np.unique(modules["cluster_id"][modules["cluster_id"] != "static"]))
            logging.info(f"identified {n_dynamic_clusters} dynamic cluster(s)")

            return sub_idx, adata_sub, delta_mean, modules


def delta_exp(
    X: np.ndarray,
    feature_names: np.ndarray = None,
    obs_names: np.ndarray = None,
    k: int = 10,
    num_subsamples: int = 1000,
    random_state: int = 0,
    n_pcs: Optional[int] = None,
    n_jobs: int = -1,
):
    """Estimates change in expression of features across representative cellular neighborhoods

    Parameters
    ----------
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    obs_names: np.ndarray (default = None)
        array containing cell names
    k: int (default = 10)
        number of nearest neighbors for between cell affinity kNN graph construction
    num_subsamples: int (default = 1000)
        number of neighborhoods to subsample when estimating feature dynamics
    random_state: int (default = 0)
        random seed parameter
    n_pcs: int (default = None)
        number of principal components for between-cell affinity graph computation.
        if None, uses adata.X to find pairwise Euclidean distances
    n_jobs: int (default = -1)
        number of tasks

    Returns
    -------
    sub_idx: np.ndarray
        array containing indices of subsampled neighborhoods
    adata_sub: anndata.AnnData
        annotated data object containing subsampled means (dimensions = num_subsamples x features)
    delta_mean: pd.DataFrame (dimensions = num_subsamples x features)
        array containing average pairwise change in expression of all features across subsampled neighborhoods
        (dimensions = num_subsamples x features)

    """
    # Construct between cell affinity kNN graph according to all profiled features
    W = construct_affinity(X=X, k=k, n_pcs=n_pcs, n_jobs=n_jobs)

    # Compute neighborhood means
    n_bool = W.astype(bool)
    n_mean = (X.transpose() @ n_bool) / np.asarray(n_bool.sum(1)).reshape(1, -1)
    n_mean = pd.DataFrame(n_mean.transpose(), index=obs_names, columns=feature_names)

    # Perform subsampling of means to get representative neighborhoods using kernel herding sketching
    adata_mean = AnnData(n_mean)
    sub_idx, adata_sub = sketch(adata_mean, num_subsamples=num_subsamples, frequency_seed=random_state, n_jobs=n_jobs)

    # Compute the average pairwise change in the expression across all neighborhoods for all features
    subsampled_means = np.asarray(adata_sub.X, dtype=np.float32)
    sub_n_obs, sub_n_vars = subsampled_means.shape

    delta_mean = subsampled_means.reshape(-1, 1, sub_n_vars) - subsampled_means.reshape(1, -1, sub_n_vars)
    delta_mean = delta_mean.sum(axis=1) * (1.0 / (sub_n_obs - 1))
    delta_mean = pd.DataFrame(
        delta_mean[np.argsort(adata_sub.obs.index)],
        index=adata_sub.obs.index[np.argsort(adata_sub.obs.index)],
        columns=adata_sub.var_names,
    )  # resort according to subsampled indices

    return sub_idx[0], adata_sub, delta_mean


def construct_affinity(X: np.ndarray, k: int = 10, radius: int = 3, n_pcs: Optional[int] = None, n_jobs: int = -1):
    """Computes between cell affinity knn graph using heat kernel

    Parameters
    ----------
    X: np.ndarray (default = None)
        Data (dimensions = cells x features)
    k: int (default = None)
        Number of nearest neighbors
    radius: int (default = 3)
        Neighbor to compute per cell distance for heat kernel bandwidth parameter
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph
        construction. If None, uses adata.X
    n_jobs: int (default = -1)
        Number of tasks
    ----------

    Returns
    -------
    W: np.ndarray
        sparse symmetric matrix containing between cell similarity (dimensions = cells x cells)
    ----------
    """
    if n_pcs is not None:
        n_comp = min(n_pcs, X.shape[1])
        pca_op = PCA(n_components=n_comp, random_state=0)
        X_ = pca_op.fit_transform(X)
    else:
        X_ = X.copy()

    # find kNN
    knn_tree = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean", n_jobs=n_jobs).fit(X_)
    dist, nn = knn_tree.kneighbors()  # dist = cells x knn (no self interactions)

    # transform distances using heat kernel
    s = heat_kernel(dist, radius=radius)  # -||x_i - x_j||^2 / 2*sigma_i**2
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = nn.reshape(-1)
    W = scipy.sparse.csr_matrix((s.reshape(-1), (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # make symmetric
    bigger = W.transpose() > W
    W = W - W.multiply(bigger) + W.transpose().multiply(bigger)

    return W


def heat_kernel(dist: np.ndarray, radius: int = 3):
    """Transforms distances into weights using heat kernel

    Parameters
    ----------
    dist: np.ndarray (default = None)
        distance matrix (dimensions = cells x k)
    radius: np.int (default = 3)
        defines the per-cell bandwidth parameter (distance to the radius nn)
    ----------

    Returns
    -------
    s: np.ndarray
        array containing between cell similarity (dimensions = cells x k)
    ----------
    """
    sigma = dist[:, [radius]]  # per cell bandwidth parameter (distance to the radius nn)
    s = np.exp(-1 * (dist**2) / (2.0 * sigma**2))  # -||x_i - x_j||^2 / 2*sigma_i**2
    return s


def _run_cluster(
    delta_mean: pd.DataFrame,
    feature_names: np.ndarray,
    n_clusters: int = 5,
    null_iterations: int = 1000,
    state: int = 0,
):
    """Multiprocessing function for identifying feature modules and performing gene-wise permutation testing

    Parameters
    ----------
    delta_mean: `pd.DataFrame`
        dataframe containing average pairwise change in expression of all features across subsampled neighborhoods
        (dimensions = num_subsamples x features)
    feature_names: np.ndarray (default = None)
        array containing feature names
    n_clusters: int (default = 5)
        number of feature modules
    null_iterations: int (default = 1000)
        number of iterations for gene-wise permutation testing
    state: int (default = 0)
        random seed parameter

    Returns
    -------
    mapping_df: `pd.DataFrame`
        dataframe containing feature to cluster assignments
    pval_df: `pd.DataFrame`
        dataframe containing the permutation p-values
    dyn_feats: np.ndarray
        array containing features identified as dynamically-expressed following permutation testing
    state: int
        random seed parameter

    """
    # perform k-means-based clustering
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=state, init="k-means++", n_init=10)
    clusters = kmeans_model.fit_predict(delta_mean.transpose())
    feats = {i: feature_names[np.where(clusters == i)[0]] for i in np.unique(clusters)}

    # record feature-cluster assignment to find intersection across runs
    mapping = np.full((len(feature_names), 1), "NaN")
    for _id, feature in feats.items():
        mapping[np.isin(feature_names, feature)] = str(_id)
    mapping_df = pd.DataFrame(mapping, index=feature_names, columns=[state])

    # compute variance-based permutation test
    seed_var = np.array(
        [np.var(delta_mean.iloc[:, np.isin(feature_names, feats[i])], axis=1, ddof=1).mean() for i in range(n_clusters)]
    )
    null_var = []
    pval_df = pd.DataFrame(index=feature_names, columns=[state])
    for f in range(0, len(feats)):
        null_var_ = np.array(
            [
                np.var(
                    delta_mean.iloc[
                        :, np.isin(feature_names, np.random.choice(feature_names, len(feats[f]), replace=False))
                    ],
                    axis=1,
                    ddof=1,
                ).mean()
                for _ in range(null_iterations)
            ]
        )
        permutation_pval = 1 - (len(np.where(seed_var[f] > null_var_)[0]) + 1) / (null_iterations + 1)
        pval_df.loc[feats[f]] = permutation_pval
        null_var.append(np.mean(null_var_))

    dynamic_id = np.where(seed_var > np.array(null_var))[0]  # select dynamic clusters over null variance threshold

    if len(dynamic_id) != 0:
        dyn_feats = np.concatenate([v for k, v in feats.items() if k in np.array(list(feats.keys()))[dynamic_id]])
        return mapping_df, pval_df, dyn_feats, state


def _annotate_clusters(
    mapping_df: pd.DataFrame = None, dyn_feats: list = None, pval_df=None, random_state_idx: int = None
):
    """Annotates clusters as dynamic or static according to feature-wise permutation testing within clusters

    Parameters
    ----------
    mapping_df: `pd.DataFrame`
        dataframe containing feature-cluster ids from KMeans clustering across random trials
        (dimensions = features x n_random_state)
    dyn_feats:  np.ndarray
        array containing features considered to be dynamically expressed across runs
    random_state_idx:  int (default = None)
        id of random state column id in mapping DataFrame to obtain cluster ids

    Returns
    -------
    modules: `pd.DataFrame`
        dataframe containing annotated feature-cluster assignment and permutation p-values (dimensions = features x 2)

    """
    cluster_id = np.unique(mapping_df.values)
    dynamic_id = np.unique(mapping_df.loc[dyn_feats].loc[:, random_state_idx])
    static_id = cluster_id[~np.isin(cluster_id, dynamic_id)]

    cats = {id_: "static" for id_ in static_id}
    cats.update({id_: f"dynamic {i}" if len(dynamic_id) > 1 else "dynamic" for i, id_ in enumerate(dynamic_id)})

    modules = pd.Categorical(pd.Series(mapping_df.loc[:, random_state_idx].astype("str")).map(cats))
    modules = pd.DataFrame(modules, index=mapping_df.index, columns=["cluster_id"])
    modules[~np.isin(modules.index, dyn_feats)] = "static"
    modules["cluster_permutation_pval"] = pval_df.median(1)  # median across all random trials
    return modules


def laplacian_score(X: np.ndarray, W: Union[np.ndarray, scipy.sparse.csr_matrix]):
    """Computes the Laplacian score :cite:`he2005laplacian`

    Parameters
    ----------
    X: np.ndarray (default = None)
        array containing normalized and preprocessed data (dimensions = cells x features)
    W: np.ndarray (default = None)
        adjacency matrix containing between-cell affinity weights

    Returns
    -------
    l_score: np.ndarray
        array containing laplacian score for all features (dimensions = features)

    """
    n_samples, n_features = X.shape

    # compute degree matrix
    D = np.array(W.sum(axis=1))
    D = scipy.sparse.diags(np.transpose(D), [0])

    # compute graph laplacian
    L = D - W.toarray()

    # ones vector: 1 = [1,···,1]'
    ones = np.ones((n_samples, n_features))

    # feature vector: fr = [fr1,...,frm]'
    fr = X.copy()

    # construct fr_t = fr - (fr' D 1/ 1' D 1) 1
    numerator = np.matmul(np.matmul(np.transpose(fr), D.toarray()), ones)
    denominator = np.matmul(np.matmul(np.transpose(ones), D.toarray()), ones)
    ratio = numerator / denominator
    ratio = ratio[:, 0]
    ratio = np.tile(ratio, (n_samples, 1))
    fr_t = fr - ratio

    # compute laplacian score Lr = fr_t' L fr_t / fr_t' D fr_t
    l_score = np.matmul(np.matmul(np.transpose(fr_t), L), fr_t) / np.matmul(
        np.dot(np.transpose(fr_t), D.toarray()), fr_t
    )
    l_score = np.diag(l_score)

    return l_score


def random_feats(X: np.ndarray, gamma: Union[int, float] = 1, frequency_seed: int = None):
    """Computes random Fourier frequency features

    Parameters
    ----------
    X: np.ndarray
        array of input data (dimensions = cells x features)
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution
    frequency_seed: int (default = None):
        random state parameter

    Returns
    -------
    phi: np.ndarray
        random Fourier frequency features (dimensions = cells x 2000)

    """
    scale = 1 / gamma

    if frequency_seed is not None:
        np.random.seed(frequency_seed)
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))
    else:
        W = np.random.normal(scale=scale, size=(X.shape[1], 1000))

    XW = np.dot(X, W)
    sin_XW = np.sin(XW)
    cos_XW = np.cos(XW)
    phi = np.concatenate((cos_XW, sin_XW), axis=1)

    return phi


def kernel_herding(phi: np.ndarray, num_subsamples: int = None):
    """Performs kernel herding subsampling

    Parameters
    ----------
    phi: np.ndarray
        random Fourier frequency features (dimensions = cells x 2000)
    num_subsamples: int (default = None)
        number of cells to subsample

    Returns
    -------
    kh_indices: np.ndarray
        indices of subsampled cells

    """
    w_t = np.mean(phi, axis=0)
    w_0 = w_t
    kh_indices = []
    while len(kh_indices) < num_subsamples:
        indices = np.argsort(np.dot(phi, w_t))[::-1]
        new_ind = next((idx for idx in indices if idx not in kh_indices), None)
        w_t = w_t + w_0 - phi[new_ind]
        kh_indices.append(new_ind)

    return kh_indices


def _parse_input(adata: AnnData):
    """Accesses and parses data from adata object

    Parameters
    ----------
    adata: AnnData
        annotated data object where adata.X is the attribute for preprocessed data

    Returns
    -------
    X: np.ndarray
        array of data (dimensions = cells x features)
    """
    X = None
    try:
        if isinstance(adata, AnnData):
            X = adata.X.copy()
        if isinstance(X, scipy.sparse.csr_matrix):
            X = np.asarray(X.todense())
        if is_numeric_dtype(adata.obs_names):
            logging.warning("Converting cell IDs to strings.")
            adata.obs_names = adata.obs_names.astype("str")
    except NameError:
        pass

    return X


def kernel_herding_main(
    sample_set_ind,
    X: np.ndarray = None,
    gamma: Union[int, float] = 1,
    frequency_seed: int = None,
    num_subsamples: int = 500,
):
    """Performs kernel herding subsampling on a single sample-set using random features

    Parameters
    ----------
    X: np.ndarray
        array of input data (dimensions = cells x features)
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution
    frequency_seed: int (default = None):
        random state parameter
    num_subsamples: int (default = None)
        number of cells to subsample
    sample_set_ind: np.ndarray
        array containing the indices of the sample-set to subsample. if you'd like to use all cells within X,
        please pass in `np.arange(0, X.shape[0])`

    Returns
    -------
    kh_indices: np.ndarray
        indices of subsampled cells within the sample-set
    """
    X = X[sample_set_ind, :]
    phi = random_feats(X, gamma=gamma, frequency_seed=frequency_seed)
    kh_indices = kernel_herding(phi, num_subsamples)

    return kh_indices


def sketch(
    adata,
    sample_set_key: str = None,
    sample_set_inds=None,
    gamma: Union[int, float] = 1,
    frequency_seed: int = None,
    num_subsamples: int = 500,
    n_jobs: int = -1,
):
    """Constructs a sketch using kernel herding and random Fourier frequency features

    Parameters
    ----------
    adata: anndata.Anndata
        annotated data object (dimensions = cells x features)
    sample_set_key: str (default = None)
        string referring to the key within `adata.obs` that contains the sample-sets to subsample
            ~ if sample_set_key is None, will parse according to sample_set_inds
            ~ if sample_set_key is None and sample_set_inds is None, will use all cells as a single sample-set
    sample_set_inds: list (default = None)
        list of arrays containing the indices of the sample-sets to subsample. (dimensions = len(sample_sets)) e.g.
        [np.array([]), np.array([]), ... , np.array([])]
            ~ if sample_set_key is None and sample_set_inds is None, will use all cells as a single sample-set
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution within random Fourier frequency feature computation
    frequency_seed: int (default = None):
        random state parameter
    num_subsamples: int (default = None)
        number of cells to subsample per sample-set
    n_jobs: int (default = -1)
        number of tasks

    Returns
    -------
    kh_indices: np.ndarray
        list of indices of subsampled cells per sample-set e.g.
        [np.array(ind0_S0...indx_S0), np.array(ind0_S1 ... indx_S1), ... , np.array(ind0_SX ... indx_SX)]
    adata_subsample: anndata.AnnData
        annotated data object containing subsampled data

    """
    import anndata

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs < -1:
        n_jobs = mp.cpu_count() + 1 + n_jobs

    if isinstance(adata, AnnData) and (sample_set_key is not None):
        sample_set_id, idx = np.unique(adata.obs[sample_set_key], return_index=True)
        sample_set_id = sample_set_id[np.argsort(idx)]
        sample_set_inds = [np.where(adata.obs[sample_set_key] == i)[0] for i in sample_set_id]
    elif sample_set_inds is None:
        sample_set_inds = [np.arange(0, adata.X.shape[0])]

    min_cell_size = min([len(i) for i in sample_set_inds])
    if num_subsamples > min_cell_size:
        warnings.warn(
            f"Number of subsamples per sample-set {num_subsamples} is greater than the maximum"
            f"number of cells in the smallest sample-set {min_cell_size}. \n"
            f"Performing subsampling using {min_cell_size} cells per sample-set",
            category=UserWarning,
            stacklevel=2,
        )
        num_subsamples = min_cell_size

    n_sample_sets = len(sample_set_inds)
    X = _parse_input(adata)

    p = mp.Pool(n_jobs)
    results = p.imap(
        partial(kernel_herding_main, X=X, gamma=gamma, frequency_seed=frequency_seed, num_subsamples=num_subsamples),
        sample_set_inds,
    )

    kh_indices = []
    with tqdm(range(n_sample_sets), desc="performing subsampling") as t:
        for result in results:
            kh_indices.append(result)
            t.update()
    p.close()
    p.join()

    adata_subsample = []
    for i in range(0, len(sample_set_inds)):
        sample_set = adata[sample_set_inds[i], :].copy()
        subsampled_sample_set = sample_set[kh_indices[i], :].copy()
        adata_subsample.append(subsampled_sample_set)

    adata_subsample = anndata.concat(adata_subsample)

    return kh_indices, adata_subsample
