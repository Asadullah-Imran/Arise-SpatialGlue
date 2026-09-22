#!/usr/bin/env python3
"""
================================================================================
  🧬 AriseSpatialGlue_4Encoder_1Layer_DEC.py

  4-Encoder 1-Layer Arise Architecture with Spatial Potts DEC:
  1. 📐 4-Stream 1-Layer GCNs (RNA Spatial, RNA Sim, Aux Spatial, Aux Sim)
  2. 🧬 RNA Dimensionality Reduction via PCA (n_comps = 60 / 100)
  3. 🔗 Static Concat Fusion (Intra-Omic + Inter-Omic Linear Fusion)
  4. 📉 Multi-Head Reconstruction Loss + Spatial Contrastive Regularization
  5. 🎯 Spatial Potts MRF-Regularized Consensus DEC (2-Stage Fine-Tuning)
  6. ⚖️ Static Loss Weights (β=25, γ=10, δ=1, κ=1.0)
  7. 📈 Complete Diagnostic Visual Analytics Suite:
     - Multi-Task Training Curves (Total Loss, Silhouette, ARI) with Stage Boundary
     - Spatial Domain Clustering Maps (Ground Truth vs. Model Predictions)
     - UMAP Scatter Plots (Ground Truth & Predicted Domains)
     - Violin Profiles: Spot Silhouette Profiles & Latent Dimension Distributions

  NOTE: This is Base Model + DEC only. No Cross-Attention, No Gram Alignment,
        No Sinkhorn OT, No Kendall & Gal Uncertainty Balancing.
================================================================================
"""

import os
import sys
import math
import time
import copy
import random
import argparse
import warnings
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from tqdm import tqdm

import scanpy as sc
import anndata as ad
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. Dataset Registry & Default Settings
# ==============================================================================

DATASET_REGISTRY = {
    "10x_human_lymph_node_A1": {
        "index": 0,
        "url": "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "10x_human_lymph_node_D1": {
        "index": 1,
        "url": "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "Mouse_Brain_E11_S1": {
        "index": 2,
        "url": "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E13_S1": {
        "index": 3,
        "url": "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E15_S1": {
        "index": 4,
        "url": "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E18_S1": {
        "index": 5,
        "url": "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
}

DATASET_LIST = list(DATASET_REGISTRY.keys())
DEFAULT_SEEDS = [42, 1234, 2024]


# ==============================================================================
# 2. Reproducibility Seeding
# ==============================================================================

def set_seed(seed: int = 42):
    """Set global random seed across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================================================================
# 3. Data Preprocessing (with RNA PCA)
# ==============================================================================

def clr_normalize_each_cell(adata, inplace=True):
    """Centered Log-Ratio (CLR) normalization on protein (ADT) data."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def pca(adata, use_reps=None, n_comps=60):
    """PCA dimensionality reduction."""
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    return feat_pca


def tfidf(X):
    """Compute TF-IDF matrix in CSR sparse format."""
    idf = X.shape[0] / (X.sum(axis=0) + 1e-10)
    if sp.issparse(X):
        tf = X.multiply(1 / (X.sum(axis=1) + 1e-10))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-10)
        return tf * idf


def preprocess_with_rna_pca(adata_RNA, adata_omics2, dataset_name: str, rna_pca_comps: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing with RNA PCA dimensionality reduction:
      - RNA: Normalized, log-transformed, HVG filtered (3000 genes), then PCA reduced to n_comps (e.g. 60 or 100).
      - ADT / ATAC: CLR + scale or TF-IDF + normalize + PCA (60 comps).
    """
    sc.pp.filter_genes(adata_RNA, min_cells=10)
    try:
        sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
        adata_RNA = adata_RNA[:, adata_RNA.var['highly_variable']].copy()
    except Exception:
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)

    rna_pca_comps = min(rna_pca_comps, adata_RNA.shape[1] - 1)
    rna_pca_features = pca(adata_RNA, n_comps=rna_pca_comps)

    cfg = DATASET_REGISTRY.get(dataset_name, {})
    ds_type = cfg.get("type", "10x")

    if ds_type == "10x":
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        aux_features = adata_omics2.X.toarray() if sp.issparse(adata_omics2.X) else np.array(adata_omics2.X)
    else:
        X_raw = adata_omics2.X.toarray() if sp.issparse(adata_omics2.X) else np.array(adata_omics2.X)
        X_tfidf = tfidf(X_raw)
        if sp.issparse(X_tfidf):
            X_tfidf = X_tfidf.toarray()
        X_tfidf = np.nan_to_num(X_tfidf)
        n_comps_aux = min(60, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        if n_comps_aux < 2:
            aux_features = X_tfidf
        else:
            pca_aux = PCA(n_components=n_comps_aux)
            aux_features = pca_aux.fit_transform(X_tfidf)

    return rna_pca_features, aux_features


# ==============================================================================
# 4. Data Containers & Graph Construction
# ==============================================================================

class Dual4GraphDataDEC(Data):
    """PyG Data container holding 4 graphs + dense spatial adjacency for Potts DEC."""
    def __init__(self, x_RNA, x_ADT,
                 sim_edge_index_rna, sim_edge_weight_rna,
                 dist_edge_index_rna, dist_edge_weight_rna,
                 sim_edge_index_aux, sim_edge_weight_aux,
                 dist_edge_index_aux, dist_edge_weight_aux,
                 spatial_adj):
        super().__init__()
        self.x_RNA = x_RNA
        self.x_ADT = x_ADT
        self.sim_edge_index_rna = sim_edge_index_rna
        self.sim_edge_weight_rna = sim_edge_weight_rna
        self.dist_edge_index_rna = dist_edge_index_rna
        self.dist_edge_weight_rna = dist_edge_weight_rna
        self.sim_edge_index_aux = sim_edge_index_aux
        self.sim_edge_weight_aux = sim_edge_weight_aux
        self.dist_edge_index_aux = dist_edge_index_aux
        self.dist_edge_weight_aux = dist_edge_weight_aux
        self.spatial_adj = spatial_adj


def build_4encoder_graphs_dec(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15):
    """
    Construct 4 graphs + dense spatial adjacency matrix for Potts DEC:
      1. RNA Similarity Graph (Cosine kNN)
      2. RNA Spatial Graph (Euclidean kNN)
      3. Aux Similarity Graph (Cosine kNN)
      4. Aux Spatial Graph (Euclidean kNN)
      5. Dense Spatial Adjacency Matrix (for Potts DEC target distribution)
    """
    num_nodes = RNA_expression.shape[0]

    # 1. Spatial KNN Distance Graph
    knn_graph = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_graph = knn_graph.maximum(knn_graph.T)

    dist_edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_graph.data, dtype=torch.float).to(device)

    # 2. RNA Similarity Graph
    sim_matrix_rna = cosine_similarity(RNA_expression)
    nbrs_rna = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(RNA_expression)
    _, indices_rna = nbrs_rna.kneighbors(RNA_expression)

    adj_rna = np.zeros_like(sim_matrix_rna, dtype=int)
    for i in range(len(RNA_expression)):
        for j in indices_rna[i][1:]:
            adj_rna[i, j] = 1
            adj_rna[j, i] = 1

    sim_edge_index_rna = torch.tensor(np.array(np.nonzero(adj_rna)), dtype=torch.long).to(device)
    sim_edge_weight_rna = torch.tensor(sim_matrix_rna[adj_rna > 0], dtype=torch.float).to(device)

    # 3. Aux Similarity Graph
    sim_matrix_aux = cosine_similarity(ADT_expression)
    nbrs_aux = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(ADT_expression)
    _, indices_aux = nbrs_aux.kneighbors(ADT_expression)

    adj_aux = np.zeros_like(sim_matrix_aux, dtype=int)
    for i in range(len(ADT_expression)):
        for j in indices_aux[i][1:]:
            adj_aux[i, j] = 1
            adj_aux[j, i] = 1

    sim_edge_index_aux = torch.tensor(np.array(np.nonzero(adj_aux)), dtype=torch.long).to(device)
    sim_edge_weight_aux = torch.tensor(sim_matrix_aux[adj_aux > 0], dtype=torch.float).to(device)

    # 4. Dense Spatial Adjacency Matrix for Potts DEC
    nbrs_spatial = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='euclidean').fit(cell_positions)
    _, indices_spatial = nbrs_spatial.kneighbors(cell_positions)
    adj_spatial = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in indices_spatial[i][1:]:
            adj_spatial[i, j] = 1.0
            adj_spatial[j, i] = 1.0

    spatial_adj = torch.tensor(adj_spatial, dtype=torch.float).to(device)

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return Dual4GraphDataDEC(
        x_RNA=x_RNA,
        x_ADT=x_ADT,
        sim_edge_index_rna=sim_edge_index_rna,
        sim_edge_weight_rna=sim_edge_weight_rna,
        dist_edge_index_rna=dist_edge_index,
        dist_edge_weight_rna=dist_edge_weight,
        sim_edge_index_aux=sim_edge_index_aux,
        sim_edge_weight_aux=sim_edge_weight_aux,
        dist_edge_index_aux=dist_edge_index,
        dist_edge_weight_aux=dist_edge_weight,
        spatial_adj=spatial_adj
    )

# ==============================================================================
# 5.5 Sinkhorn Optimal Transport Loss (V12)
# ==============================================================================

class SinkhornOptimalTransportLoss(nn.Module):
    """
    Differentiable Entropic Optimal Transport (Sinkhorn-Wasserstein distance).

    Aligns the probability distributions of RNA and Auxiliary modality
    embeddings using cosine-distance cost and entropic Sinkhorn iterations.

    V12 implementation:
        1. L2-normalize RNA and Auxiliary embeddings
        2. Construct cosine-distance cost matrix
        3. Use uniform marginal distributions
        4. Compute entropic transport kernel
        5. Solve Sinkhorn scaling iterations
        6. Return transport cost
    """

    def __init__(self, eps: float = 0.1, max_iter: int = 30):
        super().__init__()

        self.eps = eps
        self.max_iter = max_iter

    def forward(
        self,
        z_rna: torch.Tensor,
        z_aux: torch.Tensor
    ) -> torch.Tensor:

        N_rna = z_rna.shape[0]
        N_aux = z_aux.shape[0]

        # L2 normalization
        z_rna_norm = F.normalize(z_rna, p=2, dim=1)
        z_aux_norm = F.normalize(z_aux, p=2, dim=1)

        # Cosine distance:
        # C_ij = 1 - cosine_similarity(z_rna_i, z_aux_j)
        C = 1.0 - torch.mm(z_rna_norm, z_aux_norm.t())

        # Uniform marginal distributions
        mu = torch.full(
            (N_rna,),
            1.0 / N_rna,
            device=z_rna.device,
            dtype=z_rna.dtype
        )

        nu = torch.full(
            (N_aux,),
            1.0 / N_aux,
            device=z_aux.device,
            dtype=z_aux.dtype
        )

        # Entropic transport kernel
        K = torch.exp(-C / self.eps)

        # Sinkhorn scaling vectors
        u = torch.ones(
            N_rna,
            device=z_rna.device,
            dtype=z_rna.dtype
        )

        for _ in range(self.max_iter):

            v = nu / (
                torch.matmul(K.t(), u) + 1e-8
            )

            u = mu / (
                torch.matmul(K, v) + 1e-8
            )

        # Optimal transport plan
        T = u.unsqueeze(1) * K * v.unsqueeze(0)

        # Sinkhorn-Wasserstein transport cost
        return torch.sum(T * C)

# ==============================================================================
# 5. Spatial Potts MRF-Regularized Consensus DEC (V14)
# ==============================================================================

class SpatialPottsDEC(nn.Module):
    """
    Deep Embedding Clustering with Spatial Markov Random Field (Potts prior) regularization.
    Smooths target distribution P using neighborhood cluster consensus to eliminate
    salt-and-pepper spatial misclassifications.
    """
    def __init__(self, num_clusters: int, latent_dim: int, alpha: float = 1.0, lambda_spatial: float = 0.4):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.lambda_spatial = lambda_spatial
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def compute_q(self, z: torch.Tensor) -> torch.Tensor:
        """Student-t distribution soft assignment."""
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def compute_spatial_target_p(self, q: torch.Tensor, spatial_adj: torch.Tensor) -> torch.Tensor:
        """Spatially-smoothed Potts target distribution."""
        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + 1e-8)
        p_dec = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)

        adj_norm = spatial_adj / (torch.sum(spatial_adj, dim=1, keepdim=True) + 1e-8)
        spatial_consensus = torch.mm(adj_norm, q)

        p_spatial = p_dec * torch.exp(self.lambda_spatial * spatial_consensus)
        p_final = p_spatial / (torch.sum(p_spatial, dim=1, keepdim=True) + 1e-8)
        return p_final

    def forward(self, z: torch.Tensor, spatial_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.compute_q(z)
        p = self.compute_spatial_target_p(q.detach(), spatial_adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return q, kl_loss


# ==============================================================================
# 6. Model: Base 4-Encoder 1-Layer GCN + Spatial Potts DEC
# ==============================================================================

class DualGCNAll1Layer(nn.Module):
    """
    4-Encoder GCN architecture where ALL 4 streams use 1-layer GCNs:
      - RNA Spatial: 1-layer GCN (in_rna -> out_channels)
      - RNA Similarity: 1-layer GCN (in_rna -> out_channels)
      - Aux Spatial: 1-layer GCN (q -> out_channels)
      - Aux Similarity: 1-layer GCN (q -> out_channels)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, q, dropout=0.0):
        super(DualGCNAll1Layer, self).__init__()

        # --- 2 RNA Encoders (1-Layer GCNs) ---
        self.rna_sim = GCNConv(in_channels, out_channels)
        self.rna_dist = GCNConv(in_channels, out_channels)

        # --- 2 Auxiliary Encoders (1-Layer GCNs) ---
        self.aux_sim = GCNConv(q, out_channels)
        self.aux_dist = GCNConv(q, out_channels)

        # --- Arise Linear Fusion Layers ---
        self.fusion_layer1 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer_aux = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer2 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))

        # --- Arise Reconstruction Decoders ---
        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)          # Reconstruct RNA PCA
        self.deconv4 = nn.Linear(hidden_channels, q)                    # Reconstruct Aux
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels)      # Reconstruct Joint

    def forward(self, x_RNA, x_ADT,
                sim_edge_index_rna, sim_edge_weight_rna,
                dist_edge_index_rna, dist_edge_weight_rna,
                sim_edge_index_aux, sim_edge_weight_aux,
                dist_edge_index_aux, dist_edge_weight_aux):

        # 1. RNA Stream: 1-layer GCNs
        x_sim = self.rna_sim(x_RNA, sim_edge_index_rna, sim_edge_weight_rna)
        x_dist = self.rna_dist(x_RNA, dist_edge_index_rna, dist_edge_weight_rna)

        # 2. Auxiliary Stream: 1-layer GCNs
        aux_s = self.aux_sim(x_ADT, sim_edge_index_aux, sim_edge_weight_aux)
        aux_d = self.aux_dist(x_ADT, dist_edge_index_aux, dist_edge_weight_aux)

        # 3. Arise Fusion (Static Concat)
        fused_rna = self.fusion_layer1(torch.cat([x_sim, x_dist], dim=1))
        fused_aux = self.fusion_layer_aux(torch.cat([aux_s, aux_d], dim=1))
        fused_pro = self.fusion_layer2(torch.cat([fused_rna, fused_aux], dim=1))

        return x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro

    def reconstruct(self, z):
        return self.deconv2(F.relu(self.deconv1(z)))

    def reconstruct2(self, z):
        return self.deconv4(F.relu(self.deconv1(z)))

    def reconstruct3(self, z):
        return self.deconv5(F.relu(self.deconv1(z)))


class Dual4Encoder1LayerDEC(nn.Module):
    """
    Arise Model Wrapper: Base 4-Encoder 1-Layer + Spatial Potts DEC.

    Loss Formula:
      Stage 1: L_total = β * L_recon + γ * L_spatial + δ * L_reg
      Stage 2: L_total = β * L_recon + γ * L_spatial + δ * L_reg + κ * L_kl
    """
    def __init__(self, in_channels, hidden_channels, out_channels, q, num_clusters,
                 beta=25.0, gamma=10.0, delta=1.0, kl_weight=1.0,ot_weight=1.0, ot_eps=0.1,ot_max_iter=30, dropout=0.0,
                 l1_lambda=1e-4, l2_lambda=1e-3):
        super(Dual4Encoder1LayerDEC, self).__init__()
        self.gcn = DualGCNAll1Layer(in_channels, hidden_channels, out_channels, q, dropout)
        self.spatial_potts_dec = SpatialPottsDEC(num_clusters=num_clusters, latent_dim=out_channels)
        self.sinkhorn_ot = SinkhornOptimalTransportLoss(eps=ot_eps,max_iter=ot_max_iter)
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.kl_weight = kl_weight
        self.ot_weight = ot_weight
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def set_cluster_centers(self, centers_np: np.ndarray):
        """Initialize DEC cluster prototypes from KMeans centroids."""
        dev = next(self.parameters()).device
        self.spatial_potts_dec.cluster_centers.data = torch.tensor(centers_np, dtype=torch.float, device=dev)

    def forward(self, data: Dual4GraphDataDEC, compute_q: bool = False):
        """
        Forward pass. When compute_q=True (Stage 2), also computes DEC soft assignments.
        """
        x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro = self.gcn(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index_rna, data.sim_edge_weight_rna,
            data.dist_edge_index_rna, data.dist_edge_weight_rna,
            data.sim_edge_index_aux, data.sim_edge_weight_aux,
            data.dist_edge_index_aux, data.dist_edge_weight_aux
        )

        q, kl_loss = None, None
        if compute_q:
            q, kl_loss = self.spatial_potts_dec(fused_pro, data.spatial_adj)

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'aux_s': aux_s,
            'aux_d': aux_d,
            'fused_rna': fused_rna,
            'fused_aux': fused_aux,
            'fused_pro': fused_pro,
            'q': q,
            'kl_loss': kl_loss
        }

    def compute_regularization_loss(self):
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def cosine_similarity_mat(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, spatial_adj):
        """Spatial contrastive regularization using dense spatial adjacency."""
        graph_nei = spatial_adj
        graph_neg = 1.0 - graph_nei
        sim_mat = torch.sigmoid(self.cosine_similarity_mat(emb))

        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1 - sim_mat + 1e-10)).mean()

        return -(neigh_loss + neg_loss) / 2

    def compute_losses(self, data: Dual4GraphDataDEC, outputs: dict, stage: int = 1):
        """
        Compute total loss.
        Stage 1: Recon + Spatial + Reg (static weights)
        Stage 2: Recon + Spatial + Reg + DEC KL (static weights)
        """
        combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)

        # Reconstruction losses
        l_rec = F.mse_loss(combined_raw, self.gcn.reconstruct3(outputs['fused_pro']))
        l_sim = F.mse_loss(data.x_RNA, self.gcn.reconstruct(outputs['x_sim']))
        l_dist = F.mse_loss(data.x_RNA, self.gcn.reconstruct(outputs['x_dist']))
        l_aux_s = F.mse_loss(data.x_ADT, self.gcn.reconstruct2(outputs['aux_s']))
        l_aux_d = F.mse_loss(data.x_ADT, self.gcn.reconstruct2(outputs['aux_d']))
        total_recon = l_rec + l_sim + l_dist + l_aux_s + l_aux_d

        # Spatial regularization
        l_spatial = self.spatial_regularization_loss(outputs['fused_rna'], data.spatial_adj)

        # L1/L2 regularization
        reg_loss = self.compute_regularization_loss()

        # OT Loss
        l_ot = self.sinkhorn_ot(outputs['fused_rna'], outputs['fused_aux'])

        total_loss = (self.beta * total_recon +
                      self.gamma * l_spatial +
                      self.delta * reg_loss +
                      self.ot_weight * l_ot)

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_ot': l_ot.item(),
            'loss_kl': 0.0
        }

        # Stage 2: Add DEC KL loss
        if stage == 2 and outputs['kl_loss'] is not None:
            l_kl = outputs['kl_loss']
            total_loss = total_loss + self.kl_weight * l_kl
            loss_dict['loss_kl'] = l_kl.item()
            loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


# ==============================================================================
# 7. Two-Stage Training & Evaluation Engine
# ==============================================================================

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int, random_state: int = 42) -> np.ndarray:
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return kmeans.fit_predict(embeddings)


def evaluate_model(model, data):
    """Return fused joint representations."""
    model.eval()
    with torch.no_grad():
        outputs = model(data, compute_q=False)
    return outputs['fused_pro'].cpu().numpy(), outputs['fused_rna'].cpu().numpy(), outputs['fused_aux'].cpu().numpy()


def train_model_dec(
    model: Dual4Encoder1LayerDEC,
    data: Dual4GraphDataDEC,
    pretrain_epochs: int = 250,
    finetune_epochs: int = 150,
    lr: float = 1e-3,
    num_clusters: int = 10,
    true_labels: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Union[float, int, np.ndarray, List[float]]]:
    """
    Two-Stage Training Loop:
      - Stage 1: Representation Learning (Recon + Spatial, static weights)
      - Stage 2: Spatial Potts DEC Fine-Tuning (Recon + Spatial + KL, static weights)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_sil = -1.0
    best_epoch_sil = 0
    best_ari_val = -1.0
    best_epoch_ari = 0
    best_embeddings = None
    best_labels = None
    best_stage_sil = "Pre-train"
    best_stage_ari = "Pre-train"

    loss_history = []
    loss_recon_history = []
    loss_kl_history = []
    loss_ot_history = []
    epoch_sil_history = []
    epoch_ari_history = []

    # --------------------------------------------------------------------------
    # STAGE 1: Representation Learning Pre-training
    # --------------------------------------------------------------------------
    if verbose:
        print(f"\n[Stage 1/2] Representation Pre-training ({pretrain_epochs} Epochs)...")

    pbar1 = tqdm(range(pretrain_epochs), desc="Stage 1 Pretrain", disable=not verbose)
    for epoch in pbar1:
        model.train()
        optimizer.zero_grad()
        outputs = model(data, compute_q=False)
        loss, loss_dict = model.compute_losses(data, outputs, stage=1)
        loss.backward()
        optimizer.step()

        loss_history.append(loss_dict['loss_total'])
        loss_recon_history.append(loss_dict['loss_recon'])
        loss_kl_history.append(0.0)
        loss_ot_history.append(loss_dict['loss_ot'])

        # Validation clustering
        model.eval()
        with torch.no_grad():
            eval_out = model(data, compute_q=False)
            emb = eval_out['fused_pro'].detach().cpu().numpy()

        preds = cluster_embeddings(emb, num_clusters=num_clusters, random_state=seed)
        try:
            sil = float(silhouette_score(emb, preds))
        except Exception:
            sil = 0.0
        epoch_sil_history.append(sil)

        ari = 0.0
        ari_str = ""
        if true_labels is not None:
            ari = float(adjusted_rand_score(np.asarray(true_labels).astype(str), preds.astype(str)))
            epoch_ari_history.append(ari)
            ari_str = f" | ARI: {ari:.4f}"
            if ari > best_ari_val:
                best_ari_val = ari
                best_epoch_ari = epoch + 1
                best_stage_ari = "Pre-train"

        if sil > best_sil:
            best_sil = sil
            best_epoch_sil = epoch + 1
            best_embeddings = emb.copy()
            best_labels = preds.copy()
            best_stage_sil = "Pre-train"

        pbar1.set_postfix({'Loss': f"{loss.item():.4f}", 'Sil': f"{sil:.4f}", 'BestSil': f"{best_sil:.4f}"})

        if verbose and (epoch + 1) % 50 == 0:
            tqdm.write(f"Pretrain Ep {epoch + 1:3d}/{pretrain_epochs} | Total Loss: {loss.item():.4f} | Recon: {loss_dict['loss_recon']:.4f} | Sil: {sil:.4f}{ari_str} | Best Sil: {best_sil:.4f}")

    # --------------------------------------------------------------------------
    # Initialize Spatial Potts DEC Cluster Prototypes
    # --------------------------------------------------------------------------
    if verbose:
        print(f"\nInitializing Spatial Potts DEC Cluster Prototypes from Best Pre-train Representation (Epoch {best_epoch_sil})...")
    kmeans_init = KMeans(n_clusters=num_clusters, n_init=20, random_state=seed)
    kmeans_init.fit(best_embeddings)
    model.set_cluster_centers(kmeans_init.cluster_centers_)

    # --------------------------------------------------------------------------
    # STAGE 2: Spatial Potts DEC Fine-Tuning
    # --------------------------------------------------------------------------
    if verbose:
        print(f"\n[Stage 2/2] Spatial Potts DEC Fine-tuning ({finetune_epochs} Epochs)...")

    pbar2 = tqdm(range(finetune_epochs), desc="Stage 2 DEC", disable=not verbose)
    for epoch in pbar2:
        model.train()
        optimizer.zero_grad()
        outputs = model(data, compute_q=True)
        loss, loss_dict = model.compute_losses(data, outputs, stage=2)
        loss.backward()
        optimizer.step()

        curr_epoch = pretrain_epochs + epoch + 1
        loss_history.append(loss_dict['loss_total'])
        loss_recon_history.append(loss_dict['loss_recon'])
        loss_kl_history.append(loss_dict['loss_kl'])
        loss_ot_history.append(loss_dict['loss_ot'])

        # Validation clustering with DEC assignments
        model.eval()
        with torch.no_grad():
            eval_out = model(data, compute_q=True)
            emb = eval_out['fused_pro'].detach().cpu().numpy()
            q_probs = eval_out['q'].detach().cpu().numpy()
            preds = np.argmax(q_probs, axis=1)

        try:
            sil = float(silhouette_score(emb, preds))
        except Exception:
            sil = 0.0
        epoch_sil_history.append(sil)

        ari = 0.0
        ari_str = ""
        if true_labels is not None:
            ari = float(adjusted_rand_score(np.asarray(true_labels).astype(str), preds.astype(str)))
            epoch_ari_history.append(ari)
            ari_str = f" | ARI: {ari:.4f}"
            if ari > best_ari_val:
                best_ari_val = ari
                best_epoch_ari = curr_epoch
                best_stage_ari = "DEC-Potts"

        if sil > best_sil:
            best_sil = sil
            best_epoch_sil = curr_epoch
            best_embeddings = emb.copy()
            best_labels = preds.copy()
            best_stage_sil = "DEC-Potts"

        pbar2.set_postfix({'Loss': f"{loss.item():.4f}", 'OT': f"{loss_dict['loss_ot']:.4f}", 'KL': f"{loss_dict['loss_kl']:.4f}", 'Sil': f"{sil:.4f}", 'BestSil': f"{best_sil:.4f}"})

        if verbose and (epoch + 1) % 25 == 0:
            tqdm.write(f"DEC Ep {epoch + 1:3d}/{finetune_epochs} (Total {curr_epoch:3d}) | Total Loss: {loss.item():.4f} | OT: {loss_dict['loss_ot']:.4f} | KL: {loss_dict['loss_kl']:.4f} | Sil: {sil:.4f}{ari_str} | Best Sil: {best_sil:.4f}")

    last_sil = epoch_sil_history[-1] if epoch_sil_history else 0.0
    last_ari = epoch_ari_history[-1] if epoch_ari_history else 0.0

    if true_labels is not None and best_labels is not None:
        final_ari = adjusted_rand_score(true_labels, best_labels)
        final_nmi = normalized_mutual_info_score(true_labels, best_labels)
    else:
        final_ari, final_nmi = 0.0, 0.0

    if verbose:
        print("\n" + "=" * 70)
        print(" MODEL TRAINING FINISHED - SUMMARY ".center(70, "="))
        print("=" * 70)
        print(f"Best Silhouette Score         : {best_sil:.4f} (Epoch {best_epoch_sil}, Stage: {best_stage_sil})")
        print(f"ARI at Best Silhouette Epoch  : {final_ari:.4f}")
        print(f"NMI at Best Silhouette Epoch  : {final_nmi:.4f}")
        print(f"Peak ARI                      : {best_ari_val:.4f} (Epoch {best_epoch_ari}, Stage: {best_stage_ari})")
        print("-" * 70)
        print(f"Last Epoch Silhouette Score   : {last_sil:.4f}")
        print(f"Last Epoch ARI               : {last_ari:.4f}")
        print("=" * 70 + "\n")

    return {
        'best_silhouette': best_sil,
        'best_epoch_silhouette': best_epoch_sil,
        'best_stage_silhouette': best_stage_sil,
        'best_ari': best_ari_val,
        'best_epoch_ari': best_epoch_ari,
        'best_stage_ari': best_stage_ari,
        'best_embeddings': best_embeddings,
        'best_labels': best_labels,
        'best_nmi': final_nmi,
        'ari_at_best_sil': final_ari,
        'last_sil': last_sil,
        'last_ari': last_ari,
        'loss_history': loss_history,
        'loss_recon_history': loss_recon_history,
        'loss_kl_history': loss_kl_history,
        'epoch_sil_history': epoch_sil_history,
        'epoch_ari_history': epoch_ari_history,
        'loss_ot_history': loss_ot_history,
        'pretrain_epochs': pretrain_epochs,
        'finetune_epochs': finetune_epochs
    }


# ==============================================================================
# 8. Visualization Suite
# ==============================================================================

def plot_training_curves(training_results, dataset_name="Dataset", save_path=None, show=False):
    """
    Plot Loss Curve, Silhouette Score Curve, and ARI Curve across training epochs.
    Includes dashed stage boundary line between Stage 1 and Stage 2 DEC.
    """
    epochs_range = range(1, len(training_results['loss_history']) + 1)
    has_ari = len(training_results.get('epoch_ari_history', [])) > 0
    pretrain_ep = training_results.get('pretrain_epochs', 0)

    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Loss Curve
    axes[0].plot(epochs_range, training_results['loss_history'], color='#1f77b4', linewidth=2.0, label='Total Loss')
    if pretrain_ep > 0:
        axes[0].axvline(x=pretrain_ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Stage 1→2 (Ep {pretrain_ep})')
    axes[0].set_title(f'Loss Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True)

    # 2. Silhouette Score Curve
    axes[1].plot(epochs_range, training_results['epoch_sil_history'], color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    best_ep = training_results.get('best_epoch_silhouette', training_results.get('best_epoch', 0))
    best_sil = training_results.get('best_silhouette', training_results.get('best_sil', 0.0))
    if best_ep > 0:
        axes[1].axvline(x=best_ep, color='#d62728', linestyle='--', linewidth=1.5, label=f'Best Sil @ Ep {best_ep} ({best_sil:.4f})')
        axes[1].scatter([best_ep], [best_sil], color='#d62728', s=60, zorder=5)
    if pretrain_ep > 0:
        axes[1].axvline(x=pretrain_ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Stage 1→2')
    axes[1].set_title(f'Silhouette Score Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc='lower right')

    # 3. ARI Curve
    if has_ari:
        axes[2].plot(epochs_range, training_results['epoch_ari_history'], color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        if best_ep > 0 and len(training_results['epoch_ari_history']) >= best_ep:
            ari_at_best = training_results['epoch_ari_history'][best_ep - 1]
            axes[2].axvline(x=best_ep, color='#d62728', linestyle='--', linewidth=1.5, label=f'Best Sil Ep {best_ep} (ARI: {ari_at_best:.4f})')
            axes[2].scatter([best_ep], [ari_at_best], color='#d62728', s=60, zorder=5)
        if pretrain_ep > 0:
            axes[2].axvline(x=pretrain_ep, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Stage 1→2')
        axes[2].set_title(f'ARI Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(frameon=True, loc='lower right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved curves plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_visualizations(
    adata_RNA,
    results,
    dataset_name="Dataset",
    seed=42,
    rna_pca_comps=60,
    true_labels=None,
    output_dir="results_ot_dec",
    show=False
):
    """
    Generate and save complete visualizations:
    1. Training Curves (Loss, Silhouette, ARI) with Stage Boundary
    2. Ground Truth vs Predicted Spatial Domains
    3. UMAP Scatter Plots (Ground Truth & Predicted Domains)
    4. Violin Plots (Silhouette Coefficient & Latent Dim Profiles)
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    best_embeddings = results['best_embeddings']
    best_labels = results['best_labels']
    sil = results['best_silhouette']
    ari = results.get('ari_at_best_sil', results.get('best_ari', 0.0))

    # Ensure adata has necessary annotations
    if true_labels is not None:
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    elif 'ground_truth' not in adata_RNA.obs:
        adata_RNA.obs['ground_truth'] = pd.Categorical(best_labels.astype(str))

    adata_RNA.obsm['Arise_1Layer_DEC'] = best_embeddings
    adata_RNA.obs['predicted_domain'] = pd.Categorical(best_labels.astype(str))

    # 1. 📈 Plot Loss Curve, Silhouette Score Curve, and ARI Curve
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_training_curves.png")
    plot_training_curves(
        results,
        dataset_name=f"{dataset_name} (RNA PCA {rna_pca_comps} Comps, Seed {seed})",
        save_path=curve_path,
        show=show
    )

    # 2. 🗺️ Ground Truth vs Predicted Spatial Domains Plot
    spatial_coords = adata_RNA.obsm.get('spatial', None)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    if spatial_coords is not None:
        gt_cats = pd.Categorical(adata_RNA.obs['ground_truth'])
        pred_cats = pd.Categorical(adata_RNA.obs['predicted_domain'])

        axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=gt_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[0].set_title(f'Ground Truth ({dataset_name})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Spatial X')
        axes[0].set_ylabel('Spatial Y')
        axes[0].grid(False)

        axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=pred_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[1].set_title(f'Arise 4-Encoder 1-Layer + DEC Domains (ARI: {ari:.4f})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Spatial X')
        axes[1].set_ylabel('Spatial Y')
        axes[1].grid(False)
    else:
        axes[0].text(0.5, 0.5, 'No Spatial Coordinates', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No Spatial Coordinates', ha='center', va='center')

    plt.suptitle(f"Spatial Domains Comparison - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    spatial_path = os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_spatial.png")
    plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # 3. 🎨 Compute UMAP on Arise 1-Layer + DEC joint embeddings
    sc.pp.neighbors(adata_RNA, use_rep='Arise_1Layer_DEC')
    sc.tl.umap(adata_RNA)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc.pl.umap(
        adata_RNA,
        color='ground_truth',
        ax=axes[0],
        show=False,
        title='UMAP: Ground Truth Annotation'
    )
    sc.pl.umap(
        adata_RNA,
        color='predicted_domain',
        ax=axes[1],
        show=False,
        title=f'UMAP: Predicted Domains (Silhouette: {sil:.4f})'
    )
    plt.suptitle(f"UMAP Joint Representation - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    umap_path = os.path.join(plots_dir, "umap", f"{dataset_name}_seed{seed}_umap.png")
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # 4. 🎻 Violin Plots: Silhouette Coefficients & Latent Features
    sample_sil_values = silhouette_samples(best_embeddings, best_labels)
    adata_RNA.obs['silhouette_coefficient'] = sample_sil_values
    adata_RNA.obs['Latent_Dim_1'] = best_embeddings[:, 0]
    adata_RNA.obs['Latent_Dim_2'] = best_embeddings[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.violinplot(
        data=adata_RNA.obs,
        x='predicted_domain',
        y='silhouette_coefficient',
        palette='Set2',
        inner='quartile',
        ax=axes[0]
    )
    axes[0].axhline(sil, color='red', linestyle='--', label=f'Mean Sil: {sil:.4f}')
    axes[0].set_title("Silhouette Coefficient per Predicted Domain", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Domain")
    axes[0].set_ylabel("Silhouette Coefficient")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    sns.violinplot(
        data=adata_RNA.obs,
        x='predicted_domain',
        y='Latent_Dim_1',
        palette='tab10',
        inner='box',
        ax=axes[1]
    )
    axes[1].set_title("Latent Dimension 1 Distribution per Domain", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Domain")
    axes[1].set_ylabel("Latent Embedding Dim 1")
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f"Violin Plots: Cluster Profiles - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    violin_path = os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_violin.png")
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    print(f"📊 Visualizations saved to: {plots_dir}/")


# ==============================================================================
# 9. Dataset Loader
# ==============================================================================

def load_dataset(dataset_name: str, base_data_dir: str = "data") -> Tuple[ad.AnnData, ad.AnnData, np.ndarray, int]:
    """Loads dataset from local disk or downloads via gdown."""
    meta = DATASET_REGISTRY[dataset_name]
    base = os.path.join(base_data_dir, dataset_name)
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")
    other_path = os.path.join(base, meta['other_file'])
    annotation_path = os.path.join(base, meta['anno_file'])

    if not (os.path.exists(rna_path) and os.path.exists(other_path) and os.path.exists(annotation_path)):
        print(f"📥 Downloading files for {dataset_name} from Google Drive...")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{meta["url"]}" -O "{base}"')

    adata_RNA = sc.read_h5ad(rna_path)
    adata_omics2 = sc.read_h5ad(other_path)
    adata_RNA.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    gt_col = meta['gt_col']
    adata_RNA.obs['ground_truth'] = anno_df[gt_col]
    adata_omics2.obs['ground_truth'] = anno_df[gt_col]

    cell_positions = adata_RNA.obsm['spatial']
    num_clusters = adata_RNA.obs['ground_truth'].nunique()

    return adata_RNA, adata_omics2, cell_positions, num_clusters


# ==============================================================================
# 10. Experiment Runner (Python API)
# ==============================================================================

def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    rna_pca_comps: int = 60,
    epochs: int = 400,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    lr: float = 1e-3,
    beta: float = 25.0,
    gamma: float = 10.0,
    delta: float = 1.0,
    kl_weight: float = 1.0,
    ot_weight: float = 1.0,
    ot_eps: float = 0.1,
    ot_max_iter: int = 30,
    hidden_dim: int = 512,
    out_dim: int = 64,
    dropout: float = 0.0,
    device: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    output_dir: str = "results_arise_dec",
    data_dir: str = "data"
):
    """
    Run 4-Encoder 1-Layer Arise + Spatial Potts DEC experiments.
    Two-Stage: Pretrain → DEC Fine-tuning.
    """
    if device is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device
    dev = torch.device(device_str)

    # Stage 1 / Stage 2 epoch breakdown
    if pretrain_epochs is None or finetune_epochs is None:
        pretrain_epochs = int(epochs * 0.625)
        finetune_epochs = epochs - pretrain_epochs

    if datasets == "all" or datasets is None:
        selected_datasets = DATASET_LIST
    elif isinstance(datasets, (str, int)):
        selected_datasets = [datasets]
    else:
        selected_datasets = list(datasets)

    dataset_names = []
    for d in selected_datasets:
        if isinstance(d, int):
            if 0 <= d < len(DATASET_LIST):
                dataset_names.append(DATASET_LIST[d])
            else:
                raise ValueError(f"Invalid dataset index {d}.")
        elif isinstance(d, str):
            if d in DATASET_REGISTRY:
                dataset_names.append(d)
            elif d.isdigit() and 0 <= int(d) < len(DATASET_LIST):
                dataset_names.append(DATASET_LIST[int(d)])
            else:
                raise ValueError(f"Dataset '{d}' not recognized.")

    if seeds is None:
        run_seeds = DEFAULT_SEEDS
    else:
        run_seeds = list(seeds)

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "=" * 80)
    print(" 🚀 STARTING 4-ENCODER 1-LAYER + SPATIAL POTTS DEC EXPERIMENT ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets     : {dataset_names}")
    print(f"• Seeds ({len(run_seeds)})   : {run_seeds}")
    print(f"• RNA PCA Comps: {rna_pca_comps}")
    print(f"• Total Epochs : {epochs} (Pretrain: {pretrain_epochs}, DEC: {finetune_epochs}) | LR: {lr}")
    print(f"• Loss Weights : Beta={beta}, Gamma={gamma}, Delta={delta}, KL_Weight={kl_weight}")
    print(f"• Dimensions   : Hidden={hidden_dim}, Out={out_dim} | Device: {device_str}")
    print("=" * 80 + "\n")

    for dataset_name in dataset_names:
        print("\n" + "#" * 80)
        print(f" DATASET: {dataset_name} ".center(80, "#"))
        print("#" * 80)

        # Load Data
        adata_RNA, adata_omics2, cell_positions, num_clusters = load_dataset(dataset_name, base_data_dir=data_dir)
        true_labels = adata_RNA.obs['ground_truth'].values

        # Preprocess with RNA PCA
        RNA_expr, ADT_expr = preprocess_with_rna_pca(
            adata_RNA, adata_omics2, dataset_name, rna_pca_comps=rna_pca_comps
        )

        # Construct 4-Encoder Graphs + Dense Spatial Adjacency
        data = build_4encoder_graphs_dec(RNA_expr, ADT_expr, cell_positions, device=device_str)
        input_dim = data.x_RNA.shape[1]
        adt_dim = data.x_ADT.shape[1]

        dataset_results = []

        for seed in run_seeds:
            print(f"\n" + "-" * 60)
            print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
            print("-" * 60)

            set_seed(seed)

            # Initialize Model
            model = Dual4Encoder1LayerDEC(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=out_dim,
                q=adt_dim,
                num_clusters=num_clusters,
                beta=beta,
                gamma=gamma,
                delta=delta,
                kl_weight=kl_weight,
                ot_weight=ot_weight,
                ot_eps=ot_eps,
                ot_max_iter=ot_max_iter,
                dropout=dropout
            ).to(dev)

            # Train (Two-Stage)
            start_time = time.time()
            result = train_model_dec(
                model=model,
                data=data,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                lr=lr,
                num_clusters=num_clusters,
                true_labels=true_labels,
                seed=seed,
                verbose=True
            )
            runtime_sec = round(time.time() - start_time, 2)

            best_embeddings = result['best_embeddings']
            best_labels = result['best_labels']
            best_sil = result['best_silhouette']
            best_ep_sil = result['best_epoch_silhouette']
            stage_sil = result['best_stage_silhouette']

            # Compute Final Metrics at best silhouette epoch
            ari = adjusted_rand_score(true_labels, best_labels)
            nmi = normalized_mutual_info_score(true_labels, best_labels)
            ami = adjusted_mutual_info_score(true_labels, best_labels)
            chi = calinski_harabasz_score(best_embeddings, best_labels)
            dbi = davies_bouldin_score(best_embeddings, best_labels)

            last_sil = result['last_sil']
            last_ari = result['last_ari']
            total_epochs = pretrain_epochs + finetune_epochs

            print(f"Summary for {dataset_name} | Seed: {seed}")
            print(f"• Best Silhouette : {best_sil:.4f} (Epoch {best_ep_sil}, {stage_sil}) | ARI @ Best: {ari:.4f} | NMI: {nmi:.4f}")
            print(f"• Peak ARI        : {result['best_ari']:.4f} (Epoch {result['best_epoch_ari']}, {result['best_stage_ari']})")
            print(f"• Last Silhouette : {last_sil:.4f} (Epoch {total_epochs}) | Last ARI: {last_ari:.4f}")
            print(f"• Runtime         : {runtime_sec}s")

            # Visualize
            if visualize:
                plot_all_visualizations(
                    adata_RNA=adata_RNA,
                    results=result,
                    dataset_name=dataset_name,
                    seed=seed,
                    rna_pca_comps=rna_pca_comps,
                    true_labels=true_labels,
                    output_dir=output_dir,
                    show=show_plots
                )

            res_dict = {
                'dataset': dataset_name,
                'seed': seed,
                'Best_Silhouette': best_sil,
                'Best_Sil_Epoch': best_ep_sil,
                'Best_Sil_Stage': stage_sil,
                'Best_ARI': ari,
                'Best_NMI': nmi,
                'Peak_ARI': result['best_ari'],
                'Peak_ARI_Epoch': result['best_epoch_ari'],
                'Peak_ARI_Stage': result['best_stage_ari'],
                'Last_Silhouette': last_sil,
                'Last_ARI': last_ari,
                'AMI': ami,
                'CHI': chi,
                'DBI': dbi,
                'RNA_PCA_Comps': rna_pca_comps,
                'Total_Epochs': total_epochs,
                'Runtime_Sec': runtime_sec,
                'no_cluster': num_clusters
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)

        # Save per-dataset results
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"Arise4Encoder1Layer_DEC_{dataset_name}_results.csv"), index=False)

    # Save overall summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "Arise4Encoder1Layer_DEC_all_results.csv"), index=False)

    summary_cols = ['Best_ARI', 'Peak_ARI', 'Best_Silhouette', 'Last_ARI', 'Last_Silhouette', 'Best_NMI', 'Runtime_Sec']
    print("\n" + "=" * 88)
    print(" ALL EXPERIMENTS COMPLETED - DATASET MEAN SUMMARY ".center(88, "="))
    print("=" * 88)
    print(df_all.groupby('dataset')[summary_cols].mean().to_string())
    print("-" * 88)
    print(" OVERALL MEAN ACROSS ALL DATASETS & SEEDS ".center(88, "-"))
    print(df_all[summary_cols].mean().to_frame().T.to_string(index=False))
    print("=" * 88 + "\n")

    return df_all


# ==============================================================================
# 11. Command-Line Interface (CLI)
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="4-Encoder 1-Layer Arise + Spatial Potts DEC Multi-Omics Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--datasets', nargs='+', default=['all'],
        help="Datasets to run. Can be index (0 to 5), name (e.g. 10x_human_lymph_node_A1), or 'all'."
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=None,
        help="List of random seeds to evaluate (e.g. --seeds 42 2024 13)."
    )
    parser.add_argument(
        '--n_seeds', type=int, default=None,
        help="Use first N seeds from seed list."
    )
    parser.add_argument(
        '--rna_pca_comps', type=int, default=60,
        help="Number of PCA components for RNA modality (e.g. 60 or 100)."
    )
    parser.add_argument('--epochs', type=int, default=400, help="Total number of epochs")
    parser.add_argument('--pretrain_epochs', type=int, default=None, help="Explicit pretrain epochs (default: 62.5%% of total)")
    parser.add_argument('--finetune_epochs', type=int, default=None, help="Explicit DEC finetune epochs (default: remaining)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--beta', type=float, default=25.0, help="Reconstruction loss weight")
    parser.add_argument('--gamma', type=float, default=10.0, help="Spatial regularization weight")
    parser.add_argument('--delta', type=float, default=1.0, help="Weight decay regularization weight")
    parser.add_argument('--ot_weight',type=float,default=1.0,help="Sinkhorn OT loss weight")
    parser.add_argument('--ot_eps',type=float,default=0.1,help="Sinkhorn entropy regularization coefficient")
    parser.add_argument('--ot_max_iter',type=int,default=30,help="Number of Sinkhorn iterations")
    parser.add_argument('--kl_weight', type=float, default=1.0, help="DEC KL divergence loss weight (Stage 2)")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--out_dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--device', type=str, default=None, help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--visualize', action='store_true', default=True, help="Generate and save all plots")
    parser.add_argument('--no_visualize', action='store_false', dest='visualize', help="Disable visualizations")
    parser.add_argument('--show_plots', action='store_true', default=False, help="Display plots interactively")
    parser.add_argument('--output_dir', type=str, default='results_arise_dec', help="Directory to save CSV results and plots")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save/load datasets")

    cli_args = parser.parse_args()

    run_seeds = cli_args.seeds if cli_args.seeds is not None else DEFAULT_SEEDS
    if cli_args.n_seeds is not None and cli_args.n_seeds > 0:
        run_seeds = run_seeds[:cli_args.n_seeds]

    run_experiment(
        datasets=cli_args.datasets if len(cli_args.datasets) > 1 or cli_args.datasets[0] != 'all' else 'all',
        seeds=run_seeds,
        rna_pca_comps=cli_args.rna_pca_comps,
        epochs=cli_args.epochs,
        pretrain_epochs=cli_args.pretrain_epochs,
        finetune_epochs=cli_args.finetune_epochs,
        lr=cli_args.lr,
        beta=cli_args.beta,
        gamma=cli_args.gamma,
        delta=cli_args.delta,
        kl_weight=cli_args.kl_weight,
        ot_weight=cli_args.ot_weight,
        ot_eps=cli_args.ot_eps,
        ot_max_iter=cli_args.ot_max_iter,
        hidden_dim=cli_args.hidden_dim,
        out_dim=cli_args.out_dim,
        dropout=cli_args.dropout,
        device=cli_args.device,
        visualize=cli_args.visualize,
        show_plots=cli_args.show_plots,
        output_dir=cli_args.output_dir,
        data_dir=cli_args.data_dir
    )
