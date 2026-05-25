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

def build_knn(X_pca, k=20):

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )
    nn.fit(X_pca)

    distances, indices = nn.kneighbors(X_pca)

    return indices, distances

def knn_impute_weighted_rate_sparse(
    meth_matrix, knn_indices, knn_distances, 
    min_cell=2, n_jobs=-1, epsilon=1e-8
):
    
    n_cells, n_features = meth_matrix.shape
    
    def impute_cell(i):
        row = meth_matrix[i].copy()
        neighbors = knn_indices[i]
        distances = knn_distances[i]
        
        # Compute weights: inverse distance
        alpha = 0.5
        weights = 1 / (distances + epsilon) ** alpha
        weights = weights / weights.sum()
        
        for j in range(n_features):
            val = row[j]
            
            if val != -1:
                continue  # skip already observed values
            
            neighbor_vals = meth_matrix[neighbors, j]
            
            valid_mask = (neighbor_vals >= 0) & (neighbor_vals <= 1)
            valid_vals = neighbor_vals[valid_mask]
            valid_weights = weights[valid_mask]
            
            if len(valid_vals) < min_cell:
                row[j] = -1  # not enough valid neighbors, keep missing
                continue

            row[j] = np.sum(valid_vals * valid_weights) / valid_weights.sum()
        
        return row
    
    imputed_rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(impute_cell)(i) for i in range(n_cells)
    )
    
    return np.vstack(imputed_rows)

def knn_impute_weighted_parallel(meth_matrix, knn_indices, knn_distances, min_cell=2, n_jobs=-1):
    n_cells, n_features = meth_matrix.shape
    k = knn_indices.shape[1]
    
    def impute_cell(i):
        neighbors = knn_indices[i]
        distances = knn_distances[i]
        #weights = 1 / (distances + 1e-8)
        alpha = 0.5
        weights = 1 / (distances + 1e-8) ** alpha #use alpha to soften weight contradiction between distant and proximal neighbors
        weights = weights / weights.sum()
        
        row = meth_matrix[i, :].copy()
        for j in range(n_features):
            if row[j] != -1:
                continue
            
            neighbor_vals = meth_matrix[neighbors, j]
            valid_mask = neighbor_vals != -1
            valid_vals = neighbor_vals[valid_mask]
            valid_weights = weights[valid_mask]
            
            if len(valid_vals) < min_cell:
                row[j] = -1
            else:
                vote_hypo = valid_weights[valid_vals == 1].sum()
                vote_hyper = valid_weights[valid_vals == 0].sum()
                row[j] = 1 if vote_hypo > vote_hyper else 0
        return row
    
    # parallel loop
    imputed_rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(impute_cell)(i) for i in range(n_cells)
    )
    
    return np.vstack(imputed_rows)