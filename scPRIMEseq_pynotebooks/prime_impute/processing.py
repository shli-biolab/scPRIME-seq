import os
import scanpy as sc
import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import gc
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from ALLCools.mcds import MCDS
from scipy.stats import mode, pearsonr

def load_10x_rna(path):
    #import scanpy as sc
    adata = sc.read_10x_mtx(
        path,
        var_names='gene_symbols',
        cache=False
    )
    adata.var_names_make_unique()
    return adata

def preprocess_rna(adata, n_pcs=30):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)
    return adata

def categorize_methylation(df):
    """
    Rules:
        == 1  -> -1 (missing)
        >  1  -> 0  (hyper)
        <  1  -> 1  (hypo)
   """

    arr = df.values
    cat = np.empty_like(arr, dtype=np.int8)

    cat[arr == 1] = -1
    cat[arr > 1] = 0
    cat[arr < 1] = 1

    return pd.DataFrame(
        cat,
        index=df.index,
        columns=df.columns
    )

def load_mcds_sample(path,
                     data_dim="chrom20k_da_frac",
                     var_dim="chrom20k"):
    mcds_path = path
    mcds = MCDS.open(mcds_path, var_dim=var_dim)
    chr_matrix = np.asarray(mcds[data_dim])   
    chr_features = mcds[data_dim].coords[var_dim].values
    cell_ids = mcds[data_dim].coords["cell"].values
    chr_df = pd.DataFrame(
        chr_matrix.astype("float32"),
        index=cell_ids,
        columns=chr_features
    )
    chr_df = categorize_methylation(chr_df)
    chr_df = sparse.csr_matrix(chr_df.values.astype("int8"))
    adata = AnnData(X=chr_df,
                    obs=pd.DataFrame(index=cell_ids),
                    var=pd.DataFrame(index=chr_features))
    del chr_df
    gc.collect()
    return adata

def load_mcds_sample_wo_categorize(path,
                     data_dim="chrom20k_da_frac",
                     var_dim="chrom20k"):

    mcds = MCDS.open(path, var_dim=var_dim)
    chr_matrix = np.asarray(mcds[data_dim])   # drop mask & xarray ref
    chr_features = mcds[data_dim].coords[var_dim].values
    cell_ids = mcds[data_dim].coords["cell"].values
    # keep continuous values (1 = missing, >1 hyper, <1 hypo)
    chr_df = pd.DataFrame(
        chr_matrix.astype("float32"),
        index=cell_ids,
        columns=chr_features
    )
    # do NOT categorize here
    chr_sparse = sparse.csr_matrix(chr_df.values.astype("float32"))
    adata = AnnData(
        X=chr_sparse,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=chr_features)
    )
    del chr_df
    gc.collect()
    return adata