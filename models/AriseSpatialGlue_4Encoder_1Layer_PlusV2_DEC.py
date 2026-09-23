#!/usr/bin/env python3
"""
================================================================================
  🧬 AriseSpatialGlue_4Encoder_1Layer_PlusV2_DEC.py

  4-Encoder 1-Layer Arise Architecture with Synergistic Loss Functions & Spatial Potts DEC:
  1. 📐 4-Stream 1-Layer GCNs (RNA Spatial, RNA Sim, Aux Spatial, Aux Sim)
  2. 🧬 RNA Dimensionality Reduction via PCA (n_comps = 60 / 100)
  3. ⚡ Spatially-Constrained Cross-Attention Feature Exchange (Graph-Masked)
  4. 🔗 Dense Relational Gram Matrix Alignment (Frobenius Norm)
  5. 🌊 Entropic Sinkhorn Optimal Transport Relational Loss
  6. 🎯 Spatial Potts MRF-Regularized Consensus DEC (2-Stage Fine-Tuning)
  7. ⚖️ Kendall & Gal Uncertainty Multi-Task Adaptive Loss Weighting (5 Objectives)
  8. 📈 Complete Diagnostic Visual Analytics Suite:
     - Multi-Task Training Curves (Total Loss, KL Loss, Silhouette, ARI)
     - Spatial Domain Clustering Maps (Ground Truth vs. Model Predictions)
     - 4-Panel UMAP & Silhouette Diagnostic Reports (Latent Space + Knife Plot)
     - Violin Profiles: Spot Silhouette Profiles & Latent Dimension Distributions

  NOTE: Multi-Order Motif Topology (V9) is NOT included (clean standard graphs).
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
    fowlkes_mallows_score,
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
        sc.pp.scale(adata_RNA)
    except Exception:
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
        sc.pp.highly_variable_genes(adata_RNA, n_top_genes=3000)
        sc.pp.scale(adata_RNA)

    n_rna_hvg = min(3000, adata_RNA.shape[1])
    adata_RNA_hvg = adata_RNA[:, adata_RNA.var['highly_variable']].copy() if 'highly_variable' in adata_RNA.var else adata_RNA[:, :n_rna_hvg].copy()

    # PCA on RNA
    actual_rna_comps = min(rna_pca_comps, adata_RNA_hvg.shape[0] - 1, adata_RNA_hvg.shape[1])
    RNA_expression = pca(adata_RNA_hvg, n_comps=actual_rna_comps)

    # Preprocess Second Modality
    adata_omics2_copy = adata_omics2.copy()
    if dataset_name.startswith("10x"):
        adata_omics2_copy = clr_normalize_each_cell(adata_omics2_copy)
        sc.pp.scale(adata_omics2_copy)
        omics2_expression = adata_omics2_copy.X
    else:
        adata_omics2_copy.X = tfidf(adata_omics2_copy.X)
        sc.pp.normalize_per_cell(adata_omics2_copy, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2_copy)
        n_comps = min(60, adata_omics2_copy.shape[1], adata_omics2_copy.shape[0] - 1)
        adata_omics2_copy.obsm['feat'] = pca(adata_omics2_copy, n_comps=n_comps)
        omics2_expression = adata_omics2_copy.obsm['feat']

    if sp.issparse(omics2_expression):
        omics2_expression = omics2_expression.toarray()
    if sp.issparse(RNA_expression):
        RNA_expression = RNA_expression.toarray()

    return RNA_expression, omics2_expression


# ==============================================================================
# 4. 4-Encoder Graph Construction & Spatial Mask
# ==============================================================================

class Dual4GraphPlusData:
    """PyG Data container holding 4 graphs, dense spatial adjacency, and spatial mask."""
    def __init__(
        self,
        x_RNA: torch.Tensor,
        x_ADT: torch.Tensor,
        sim_edge_index_rna: torch.Tensor,
        sim_edge_weight_rna: torch.Tensor,
        dist_edge_index_rna: torch.Tensor,
        dist_edge_weight_rna: torch.Tensor,
        sim_edge_index_aux: torch.Tensor,
        sim_edge_weight_aux: torch.Tensor,
        dist_edge_index_aux: torch.Tensor,
        dist_edge_weight_aux: torch.Tensor,
        spatial_adj: torch.Tensor,
        spatial_mask: torch.Tensor
    ):
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
        self.spatial_mask = spatial_mask


def build_4encoder_plus_graphs(
    x_rna: np.ndarray,
    x_aux: np.ndarray,
    cell_positions: np.ndarray,
    device: str = 'cpu',
    num_neighbors: int = 15
) -> Dual4GraphPlusData:
    """
    Constructs 4 distinct graph streams with spatial masks (without motif power expansions):
      1. RNA Similarity Graph (Cosine kNN)
      2. RNA Spatial Graph (Euclidean kNN)
      3. Aux Similarity Graph (Cosine kNN)
      4. Aux Spatial Graph (Euclidean kNN)
      5. Dense Spatial Adjacency Matrix & Attention Mask
    """
    num_nodes = x_rna.shape[0]

    # 1. RNA Similarity Graph (Cosine kNN)
    nbrs_rna = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(x_rna)
    _, indices_rna = nbrs_rna.kneighbors(x_rna)
    adj_sim_rna = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in indices_rna[i][1:]:
            adj_sim_rna[i, j] = 1.0
            adj_sim_rna[j, i] = 1.0
    sim_sparse_rna = sp.csr_matrix(adj_sim_rna)
    sim_nz_rna = sim_sparse_rna.nonzero()
    sim_edge_index_rna = torch.tensor(np.array(sim_nz_rna), dtype=torch.long).to(device)
    sim_edge_weight_rna = torch.tensor(np.array(sim_sparse_rna[sim_nz_rna]).flatten(), dtype=torch.float).to(device)

    # 2. Aux Similarity Graph (Cosine kNN)
    nbrs_aux = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(x_aux)
    _, indices_aux = nbrs_aux.kneighbors(x_aux)
    adj_sim_aux = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in indices_aux[i][1:]:
            adj_sim_aux[i, j] = 1.0
            adj_sim_aux[j, i] = 1.0
    sim_sparse_aux = sp.csr_matrix(adj_sim_aux)
    sim_nz_aux = sim_sparse_aux.nonzero()
    sim_edge_index_aux = torch.tensor(np.array(sim_nz_aux), dtype=torch.long).to(device)
    sim_edge_weight_aux = torch.tensor(np.array(sim_sparse_aux[sim_nz_aux]).flatten(), dtype=torch.float).to(device)

    # 3. Spatial Distance Graph (Euclidean kNN)
    knn_graph = kneighbors_graph(cell_positions, n_neighbors=6, include_self=True)
    dist_nz = knn_graph.nonzero()
    dist_edge_index = torch.tensor(np.array(dist_nz), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(np.array(knn_graph[dist_nz]).flatten(), dtype=torch.float).to(device)

    # 4. Dense Spatial Adjacency Matrix & Mask for Cross-Attention & Potts DEC
    nbrs_spatial = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='euclidean').fit(cell_positions)
    _, indices_spatial = nbrs_spatial.kneighbors(cell_positions)
    adj_spatial = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in indices_spatial[i][1:]:
            adj_spatial[i, j] = 1.0
            adj_spatial[j, i] = 1.0

    spatial_adj = torch.tensor(adj_spatial, dtype=torch.float).to(device)
    spatial_mask = torch.tensor((adj_spatial > 0) | np.eye(num_nodes, dtype=bool), dtype=torch.float).to(device)

    x_RNA_tensor = torch.tensor(x_rna, dtype=torch.float).to(device)
    x_ADT_tensor = torch.tensor(x_aux, dtype=torch.float).to(device)

    return Dual4GraphPlusData(
        x_RNA=x_RNA_tensor,
        x_ADT=x_ADT_tensor,
        sim_edge_index_rna=sim_edge_index_rna,
        sim_edge_weight_rna=sim_edge_weight_rna,
        dist_edge_index_rna=dist_edge_index,
        dist_edge_weight_rna=dist_edge_weight,
        sim_edge_index_aux=sim_edge_index_aux,
        sim_edge_weight_aux=sim_edge_weight_aux,
        dist_edge_index_aux=dist_edge_index,
        dist_edge_weight_aux=dist_edge_weight,
        spatial_adj=spatial_adj,
        spatial_mask=spatial_mask
    )


# ==============================================================================
# 5. Spatial Graph-Masked Cross-Attention (V11)
# ==============================================================================

class GraphMaskedCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with Spatial Masking:
    Restricts attention weights to physical spatial neighbors, eliminating cross-modal noise.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_query: torch.Tensor, x_key_value: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
        N, D = x_query.shape
        residual = x_query

        Q = self.q_proj(x_query).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_key_value).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_key_value).view(N, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_bool = (spatial_mask == 0).unsqueeze(0)
        scores = scores.masked_fill(mask_bool, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(0, 1).contiguous().view(N, D)
        out = self.out_proj(context)

        return self.layer_norm(residual + out)


# ==============================================================================
# 6. Sinkhorn Entropic Optimal Transport Loss (V12)
# ==============================================================================

class SinkhornOptimalTransportLoss(nn.Module):
    """
    Differentiable Entropic Optimal Transport (Sinkhorn-Wasserstein distance)
    aligning the geometric probability manifolds across RNA and Auxiliary modalities.
    """
    def __init__(self, eps: float = 0.1, max_iter: int = 30):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        N = z_rna.shape[0]
        z_rna_norm = F.normalize(z_rna, p=2, dim=1)
        z_aux_norm = F.normalize(z_aux, p=2, dim=1)

        C = 1.0 - torch.mm(z_rna_norm, z_aux_norm.t())
        mu = torch.full((N,), 1.0 / N, device=z_rna.device, dtype=z_rna.dtype)
        nu = torch.full((N,), 1.0 / N, device=z_aux.device, dtype=z_aux.dtype)

        K = torch.exp(-C / self.eps)
        u = torch.ones(N, device=z_rna.device, dtype=z_rna.dtype)

        for _ in range(self.max_iter):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)

        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        return torch.sum(T * C)


# ==============================================================================
# 7. Spatial Potts MRF-Regularized Consensus DEC (V14)
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
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def compute_spatial_target_p(self, q: torch.Tensor, spatial_adj: torch.Tensor) -> torch.Tensor:
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
# 8. Model Architecture: 4-Encoder 1-Layer GCN + Synergistic Loss + DEC
# ==============================================================================

class Dual4Encoder1LayerPlusV2DEC(nn.Module):
    """
    🧬 4-Encoder 1-Layer Architecture with Synergistic Multi-Task Loss & Spatial Potts DEC:
      - 4 Streams of 1-Layer GCNs (RNA Spatial, RNA Sim, Aux Spatial, Aux Sim)
      - Spatially-Constrained Cross-Attention (V11)
      - Intra-omic and Inter-omic Linear Fusion
      - Multi-Head Reconstruction Decoders
      - Dense Relational Gram Matrix Alignment (V7)
      - Sinkhorn Entropic Optimal Transport Loss (V12)
      - Spatial Potts MRF-Regularized Consensus DEC (V14)
      - Kendall & Gal Adaptive Uncertainty Multi-Task Loss Balancing (5 Objectives)
    """
    def __init__(
        self,
        in_rna_dim: int,
        in_aux_dim: int,
        num_clusters: int,
        hidden_dim: int = 512,
        out_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_rna_dim = in_rna_dim
        self.in_aux_dim = in_aux_dim
        self.num_clusters = num_clusters
        self.out_dim = out_dim

        # --- 4 Streams: 1-Layer GCN Encoders ---
        self.rna_sim = GCNConv(in_rna_dim, out_dim)
        self.rna_dist = GCNConv(in_rna_dim, out_dim)
        self.aux_sim = GCNConv(in_aux_dim, out_dim)
        self.aux_dist = GCNConv(in_aux_dim, out_dim)

        # --- Intra-Omic Fusion ---
        self.fusion_rna = nn.Sequential(nn.Linear(2 * out_dim, out_dim))
        self.fusion_aux = nn.Sequential(nn.Linear(2 * out_dim, out_dim))

        # --- Spatially-Constrained Cross-Attention (V11) ---
        self.cross_attn = GraphMaskedCrossAttention(d_model=out_dim, n_heads=4, dropout=0.1)

        # --- Joint Inter-Omic Fusion ---
        self.fusion_joint = nn.Sequential(nn.Linear(2 * out_dim, out_dim))

        # --- Reconstruction Decoders ---
        self.dropout = dropout
        self.deconv1 = nn.Linear(out_dim, hidden_dim)
        self.deconv_rna = nn.Linear(hidden_dim, in_rna_dim)
        self.deconv_aux = nn.Linear(hidden_dim, in_aux_dim)
        self.deconv_joint = nn.Linear(hidden_dim, in_rna_dim + in_aux_dim)

        # --- Geometric Alignment Objectives ---
        self.ot_loss = SinkhornOptimalTransportLoss(eps=0.1, max_iter=30)
        self.spatial_potts_dec = SpatialPottsDEC(num_clusters=num_clusters, latent_dim=out_dim, lambda_spatial=0.4)

        # --- Kendall & Gal Homoscedastic Multi-Task Uncertainty Parameters (s_0..s_4) ---
        # Task losses: [L_recon, L_spatial, L_dense, L_ot, L_dec]
        self.log_vars = nn.Parameter(torch.zeros(5))

    def set_cluster_centers(self, centers_np: np.ndarray):
        dev = next(self.parameters()).device
        self.spatial_potts_dec.cluster_centers.data = torch.tensor(centers_np, dtype=torch.float, device=dev)

    def forward(self, data: Dual4GraphPlusData, compute_q: bool = False) -> Dict[str, torch.Tensor]:
        # 1. RNA Stream (1-layer GCNs)
        x_sim = self.rna_sim(data.x_RNA, data.sim_edge_index_rna, data.sim_edge_weight_rna)
        x_dist = self.rna_dist(data.x_RNA, data.dist_edge_index_rna, data.dist_edge_weight_rna)

        # 2. Aux Stream (1-layer GCNs)
        aux_s = self.aux_sim(data.x_ADT, data.sim_edge_index_aux, data.sim_edge_weight_aux)
        aux_d = self.aux_dist(data.x_ADT, data.dist_edge_index_aux, data.dist_edge_weight_aux)

        # 3. Intra-Omic Fusion
        fused_rna = self.fusion_rna(torch.cat([x_sim, x_dist], dim=1))
        fused_aux = self.fusion_aux(torch.cat([aux_s, aux_d], dim=1))

        # 4. Spatially-Masked Cross-Attention (V11)
        z_rna_refined = self.cross_attn(fused_rna, fused_aux, data.spatial_mask)

        # 5. Joint Inter-Omic Representation
        fused_joint = self.fusion_joint(torch.cat([z_rna_refined, fused_aux], dim=1))

        # 6. Spatial Potts Consensus DEC (Stage 2)
        q, kl_loss = None, None
        if compute_q:
            q, kl_loss = self.spatial_potts_dec(fused_joint, data.spatial_adj)

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'aux_s': aux_s,
            'aux_d': aux_d,
            'fused_rna': z_rna_refined,
            'fused_aux': fused_aux,
            'fused_joint': fused_joint,
            'embedding': fused_joint,
            'q': q,
            'kl_loss': kl_loss
        }

    def compute_loss(self, data: Dual4GraphPlusData, outputs: Dict[str, torch.Tensor], stage: int = 1) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_nodes = data.x_RNA.shape[0]

        # 1. Multi-Head Reconstruction Loss
        h_joint = F.relu(self.deconv1(outputs['fused_joint']))
        l_rec = F.mse_loss(torch.cat([data.x_RNA, data.x_ADT], dim=1), self.deconv_joint(h_joint))
        l_sim = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_sim']))))
        l_dist = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_dist']))))
        l_aux_s = F.mse_loss(data.x_ADT, self.deconv_aux(F.relu(self.deconv1(outputs['aux_s']))))
        l_aux_d = F.mse_loss(data.x_ADT, self.deconv_aux(F.relu(self.deconv1(outputs['aux_d']))))
        total_recon = l_rec + l_sim + l_dist + l_aux_s + l_aux_d

        # 2. Spatial Regularization Contrastive Loss
        graph_nei = data.spatial_adj
        graph_neg = 1.0 - graph_nei
        emb_norm = F.normalize(outputs['fused_rna'], p=2, dim=1, eps=1e-8)
        sim_mat = torch.sigmoid(torch.matmul(emb_norm, emb_norm.T) - torch.diag_embed(torch.diag(torch.matmul(emb_norm, emb_norm.T))))
        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_mat + 1e-10)).mean()
        l_spatial = -(neigh_loss + neg_loss) / 2.0

        # 3. Dense Cross-Modal Relational Gram Alignment Loss (V7)
        l_emb = torch.mean((outputs['fused_joint'] - outputs['fused_rna']) ** 2) + torch.mean((outputs['fused_joint'] - outputs['fused_aux']) ** 2)
        gram_r = torch.mm(outputs['fused_rna'], outputs['fused_rna'].t())
        gram_a = torch.mm(outputs['fused_aux'], outputs['fused_aux'].t())
        l_rel = torch.norm(gram_r - gram_a, p='fro') / (num_nodes * num_nodes)
        l_dense = l_emb + 0.1 * l_rel

        # 4. Sinkhorn Entropic Optimal Transport Loss (V12)
        l_ot = self.ot_loss(outputs['fused_rna'], outputs['fused_aux'])

        # Uncertainty Precision Weights (exp(-s_m))
        p0 = torch.exp(-self.log_vars[0])
        p1 = torch.exp(-self.log_vars[1])
        p2 = torch.exp(-self.log_vars[2])
        p3 = torch.exp(-self.log_vars[3])
        p4 = torch.exp(-self.log_vars[4])

        total_loss = (0.5 * p0 * total_recon + 0.5 * self.log_vars[0] +
                      0.5 * p1 * l_spatial + 0.5 * self.log_vars[1] +
                      0.5 * p2 * l_dense + 0.5 * self.log_vars[2] +
                      0.5 * p3 * l_ot + 0.5 * self.log_vars[3])

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_dense': l_dense.item(),
            'loss_ot': l_ot.item(),
            'loss_kl': 0.0
        }

        # 5. Spatial Potts Consensus DEC KL Loss (V14, Stage 2)
        if stage == 2 and outputs['kl_loss'] is not None:
            l_kl = outputs['kl_loss']
            total_loss = total_loss + 0.5 * p4 * l_kl + 0.5 * self.log_vars[4]
            loss_dict['loss_kl'] = l_kl.item()
            loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


# ==============================================================================
# 9. Two-Stage Training & Evaluation Engine
# ==============================================================================

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int, random_state: int = 42) -> np.ndarray:
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return kmeans.fit_predict(embeddings)


def train_model_plus_v2_dec(
    model: Dual4Encoder1LayerPlusV2DEC,
    data: Dual4GraphPlusData,
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
      - Stage 1: Representation Learning (Multi-Task Kendall & Gal Balancing)
      - Stage 2: Spatial Potts Consensus DEC Fine-Tuning
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
    loss_spatial_history = []
    loss_dense_history = []
    loss_ot_history = []
    loss_kl_history = []
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
        loss, loss_dict = model.compute_loss(data, outputs, stage=1)
        loss.backward()
        optimizer.step()

        loss_history.append(loss_dict['loss_total'])
        loss_recon_history.append(loss_dict['loss_recon'])
        loss_spatial_history.append(loss_dict['loss_spatial'])
        loss_dense_history.append(loss_dict['loss_dense'])
        loss_ot_history.append(loss_dict['loss_ot'])
        loss_kl_history.append(0.0)

        # Validation clustering
        model.eval()
        with torch.no_grad():
            eval_out = model(data, compute_q=False)
            emb = eval_out['fused_joint'].detach().cpu().numpy()

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
    # STAGE 2: Spatial Potts Consensus DEC Fine-Tuning (V14)
    # --------------------------------------------------------------------------
    if verbose:
        print(f"\n[Stage 2/2] Spatial Potts Consensus DEC Fine-tuning ({finetune_epochs} Epochs)...")

    pbar2 = tqdm(range(finetune_epochs), desc="Stage 2 DEC", disable=not verbose)
    for epoch in pbar2:
        model.train()
        optimizer.zero_grad()
        outputs = model(data, compute_q=True)
        loss, loss_dict = model.compute_loss(data, outputs, stage=2)
        loss.backward()
        optimizer.step()

        curr_epoch = pretrain_epochs + epoch + 1
        loss_history.append(loss_dict['loss_total'])
        loss_recon_history.append(loss_dict['loss_recon'])
        loss_spatial_history.append(loss_dict['loss_spatial'])
        loss_dense_history.append(loss_dict['loss_dense'])
        loss_ot_history.append(loss_dict['loss_ot'])
        loss_kl_history.append(loss_dict['loss_kl'])

        # Validation clustering with DEC assignments
        model.eval()
        with torch.no_grad():
            eval_out = model(data, compute_q=True)
            emb = eval_out['fused_joint'].detach().cpu().numpy()
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

        pbar2.set_postfix({'Loss': f"{loss.item():.4f}", 'KL': f"{loss_dict['loss_kl']:.4f}", 'Sil': f"{sil:.4f}", 'BestSil': f"{best_sil:.4f}"})

        if verbose and (epoch + 1) % 25 == 0:
            tqdm.write(f"DEC Ep {epoch + 1:3d}/{finetune_epochs} (Total {curr_epoch:3d}) | Total Loss: {loss.item():.4f} | KL: {loss_dict['loss_kl']:.4f} | Sil: {sil:.4f}{ari_str} | Best Sil: {best_sil:.4f}")

    return {
        'best_silhouette': best_sil,
        'best_epoch_silhouette': best_epoch_sil,
        'best_stage_silhouette': best_stage_sil,
        'best_ari': best_ari_val,
        'best_epoch_ari': best_epoch_ari,
        'best_stage_ari': best_stage_ari,
        'best_embeddings': best_embeddings,
        'best_labels': best_labels,
        'loss_history': loss_history,
        'loss_recon_history': loss_recon_history,
        'loss_spatial_history': loss_spatial_history,
        'loss_dense_history': loss_dense_history,
        'loss_ot_history': loss_ot_history,
        'loss_kl_history': loss_kl_history,
        'epoch_sil_history': epoch_sil_history,
        'epoch_ari_history': epoch_ari_history,
        'pretrain_epochs': pretrain_epochs,
        'finetune_epochs': finetune_epochs
    }


# ==============================================================================
# 10. Comprehensive Metric Evaluation
# ==============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes full suite of spatial domain clustering metrics."""
    y_true_str = np.asarray(y_true).astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)

    ari = float(adjusted_rand_score(y_true_str, y_pred_str))
    nmi = float(normalized_mutual_info_score(y_true_str, y_pred_str))
    ami = float(adjusted_mutual_info_score(y_true_str, y_pred_str))
    homo = float(homogeneity_score(y_true_str, y_pred_str))
    v_meas = float(v_measure_score(y_true_str, y_pred_str))
    fmi = float(fowlkes_mallows_score(y_true_str, y_pred_str))

    sil, ch_score, db_score = 0.0, 0.0, 0.0
    if embeddings is not None and len(np.unique(y_pred_str)) > 1:
        try:
            sil = float(silhouette_score(embeddings, y_pred_str))
        except Exception:
            sil = 0.0
        try:
            ch_score = float(calinski_harabasz_score(embeddings, y_pred_str))
        except Exception:
            ch_score = 0.0
        try:
            db_score = float(davies_bouldin_score(embeddings, y_pred_str))
        except Exception:
            db_score = 0.0

    return {
        'ARI': ari,
        'NMI': nmi,
        'AMI': ami,
        'Homo': homo,
        'V-Measure': v_meas,
        'FMI': fmi,
        'Silhouette': sil,
        'Calinski_Harabasz': ch_score,
        'Davies_Bouldin': db_score
    }


# ==============================================================================
# 11. Complete Visual Analytics Suite
# ==============================================================================

def plot_training_curves(
    results: dict,
    dataset_name: str,
    save_path: Optional[str] = None,
    show: bool = False
):
    """
    Plots multi-task loss trajectories, silhouette score progression, and ARI curve,
    with a dashed indicator marking the Stage 1 / Stage 2 DEC boundary.
    """
    epochs_total = len(results['loss_history'])
    epochs_range = list(range(1, epochs_total + 1))
    pretrain_ep = results.get('pretrain_epochs', int(epochs_total * 0.625))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_theme(style="whitegrid")

    # 1. Total & Task Losses
    axes[0].plot(epochs_range, results['loss_history'], label='Total Loss', color='#1f77b4', lw=2)
    if 'loss_recon_history' in results:
        axes[0].plot(epochs_range, results['loss_recon_history'], label='Recon', color='#ff7f0e', lw=1.2, alpha=0.7)
    if 'loss_spatial_history' in results:
        axes[0].plot(epochs_range, results['loss_spatial_history'], label='Spatial', color='#2ca02c', lw=1.2, alpha=0.7)
    if 'loss_dense_history' in results:
        axes[0].plot(epochs_range, results['loss_dense_history'], label='Gram Align', color='#9467bd', lw=1.2, alpha=0.7)
    if 'loss_ot_history' in results:
        axes[0].plot(epochs_range, results['loss_ot_history'], label='Sinkhorn OT', color='#8c564b', lw=1.2, alpha=0.7)
    if 'loss_kl_history' in results:
        axes[0].plot(epochs_range, results['loss_kl_history'], label='Potts DEC KL', color='#d62728', lw=1.5, ls='--')

    axes[0].axvline(x=pretrain_ep, color='#555555', linestyle=':', label='DEC Fine-tune Start')
    axes[0].set_title(f"Multi-Task Loss Curves\n{dataset_name}", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2. Silhouette Score Trajectory
    axes[1].plot(epochs_range, results['epoch_sil_history'], label='Silhouette Score', color='#2ca02c', lw=2)
    best_ep_sil = results['best_epoch_silhouette']
    best_val_sil = results['best_silhouette']
    stage_sil = results.get('best_stage_silhouette', 'Pre-train')
    axes[1].scatter([best_ep_sil], [best_val_sil], color='red', s=70, zorder=5,
                    label=f'Best Sil: {best_val_sil:.4f} (Ep {best_ep_sil}, {stage_sil})')
    axes[1].axvline(x=pretrain_ep, color='#555555', linestyle=':', label='DEC Start')
    axes[1].set_title(f"Silhouette Score Curve\n{dataset_name}", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Silhouette Score", fontsize=10)
    axes[1].legend(loc="lower right", fontsize=8)

    # 3. ARI Trajectory
    if results.get('epoch_ari_history') and len(results['epoch_ari_history']) > 0:
        axes[2].plot(epochs_range, results['epoch_ari_history'], label='ARI Score', color='#e377c2', lw=2)
        best_ep_ari = results['best_epoch_ari']
        best_val_ari = results['best_ari']
        stage_ari = results.get('best_stage_ari', 'Pre-train')
        axes[2].scatter([best_ep_ari], [best_val_ari], color='red', s=80, marker='*', zorder=5,
                        label=f'Peak ARI: {best_val_ari:.4f} (Ep {best_ep_ari}, {stage_ari})')
        if best_ep_sil <= len(results['epoch_ari_history']):
            sil_ari = results['epoch_ari_history'][best_ep_sil - 1]
            axes[2].scatter([best_ep_sil], [sil_ari], color='#1f77b4', s=60, marker='o', zorder=5,
                            label=f'ARI @ Best Sil: {sil_ari:.4f}')
        axes[2].axvline(x=pretrain_ep, color='#555555', linestyle=':', label='DEC Start')
        axes[2].set_title(f"Adjusted Rand Index (ARI) Curve\n{dataset_name}", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Epoch", fontsize=10)
        axes[2].set_ylabel("ARI", fontsize=10)
        axes[2].legend(loc="lower right", fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "ARI trajectory not available", ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_spatial_domains(
    cell_positions: np.ndarray,
    ground_truth: np.ndarray,
    predicted_domain: np.ndarray,
    dataset_name: str,
    ari: float,
    save_path: Optional[str] = None,
    show: bool = False
):
    """2D spatial scatter plot comparing Ground Truth vs. Model Predicted Domains."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    gt_labels = np.asarray(ground_truth).astype(str)
    pred_labels = np.asarray(predicted_domain).astype(str)
    unique_gt = np.unique(gt_labels)
    unique_pred = np.unique(pred_labels)

    palette_gt = sns.color_palette("tab20", len(unique_gt))
    palette_pred = sns.color_palette("tab20", len(unique_pred))
    color_map_gt = {val: palette_gt[i] for i, val in enumerate(unique_gt)}
    color_map_pred = {val: palette_pred[i] for i, val in enumerate(unique_pred)}

    for val in unique_gt:
        idx = (gt_labels == val)
        axes[0].scatter(cell_positions[idx, 0], cell_positions[idx, 1],
                        c=[color_map_gt[val]], label=str(val), s=15, alpha=0.85, edgecolors='none')
    axes[0].set_title(f'Ground Truth ({dataset_name})', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    if len(unique_gt) <= 15:
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=8, markerscale=2)

    for val in unique_pred:
        idx = (pred_labels == val)
        axes[1].scatter(cell_positions[idx, 0], cell_positions[idx, 1],
                        c=[color_map_pred[val]], label=f"Domain {val}", s=15, alpha=0.85, edgecolors='none')
    axes[1].set_title(f'Arise 4-Encoder 1-Layer + PlusV2 + DEC (ARI: {ari:.4f})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    if len(unique_pred) <= 15:
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=8, markerscale=2)

    plt.suptitle(f"Spatial Domains Comparison - {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🗺️  Spatial domains plot saved to -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_umap_and_silhouette(
    embeddings: np.ndarray,
    ground_truth: np.ndarray,
    predicted_domain: np.ndarray,
    dataset_name: str,
    ari: float,
    overall_sil: float,
    save_path: Optional[str] = None,
    show: bool = False
):
    """4-Panel UMAP & Silhouette Diagnostic Suite (UMAP GT, UMAP Pred, Silhouette Heatmap & Knife Profile)."""
    adata_temp = ad.AnnData(X=embeddings)
    sc.pp.neighbors(adata_temp, use_rep='X', n_neighbors=15)
    sc.tl.umap(adata_temp)
    umap_coords = adata_temp.obsm['X_umap']

    gt_labels = np.asarray(ground_truth).astype(str)
    pred_labels = np.asarray(predicted_domain).astype(str)
    sample_sils = silhouette_samples(embeddings, pred_labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: UMAP Ground Truth
    unique_gt = np.unique(gt_labels)
    palette_gt = sns.color_palette("tab20", len(unique_gt))
    for i, val in enumerate(unique_gt):
        idx = (gt_labels == val)
        axes[0, 0].scatter(umap_coords[idx, 0], umap_coords[idx, 1],
                           c=[palette_gt[i]], label=str(val), s=15, alpha=0.85, edgecolors='none')
    axes[0, 0].set_title(f"UMAP - Ground Truth ({dataset_name})", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("UMAP 1")
    axes[0, 0].set_ylabel("UMAP 2")
    if len(unique_gt) <= 15:
        axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=2)

    # Panel 2: UMAP Predicted Domains
    unique_pred = np.unique(pred_labels)
    palette_pred = sns.color_palette("tab20", len(unique_pred))
    for i, val in enumerate(unique_pred):
        idx = (pred_labels == val)
        axes[0, 1].scatter(umap_coords[idx, 0], umap_coords[idx, 1],
                           c=[palette_pred[i]], label=f"Domain {val}", s=15, alpha=0.85, edgecolors='none')
    axes[0, 1].set_title(f"UMAP - Model Predictions (ARI: {ari:.4f})", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("UMAP 1")
    axes[0, 1].set_ylabel("UMAP 2")
    if len(unique_pred) <= 15:
        axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=2)

    # Panel 3: UMAP Colored by Sample Silhouette Score
    scatter = axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                                 c=sample_sils, cmap='coolwarm', s=15, alpha=0.85, edgecolors='none')
    axes[1, 0].set_title(f"UMAP - Spot Silhouette Coefficients (Mean: {overall_sil:.4f})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("UMAP 1")
    axes[1, 0].set_ylabel("UMAP 2")
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label("Silhouette Coefficient", fontsize=10)

    # Panel 4: Silhouette Knife Profile Plot
    y_lower = 10
    for i, val in enumerate(unique_pred):
        cluster_sils = sample_sils[pred_labels == val]
        cluster_sils.sort()
        size_cluster = cluster_sils.shape[0]
        y_upper = y_lower + size_cluster
        axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sils,
                                facecolor=palette_pred[i], edgecolor=palette_pred[i], alpha=0.7)
        axes[1, 1].text(-0.05, y_lower + 0.5 * size_cluster, f"D{val}", fontsize=8, fontweight='bold')
        y_lower = y_upper + 10

    axes[1, 1].axvline(x=overall_sil, color="red", linestyle="--", lw=1.5, label=f'Mean Sil ({overall_sil:.4f})')
    axes[1, 1].set_title("Silhouette Knife Profile Plot", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Silhouette Coefficient Values")
    axes[1, 1].set_ylabel("Cluster Label (Domain)")
    axes[1, 1].set_yticks([])
    axes[1, 1].legend(loc="lower right", fontsize=8)

    plt.suptitle(f"Latent Space & Silhouette Diagnostics - {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 UMAP & Silhouette diagnostics saved to -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_violin_profiles(
    embeddings: np.ndarray,
    predicted_domain: np.ndarray,
    dataset_name: str,
    save_path: Optional[str] = None,
    show: bool = False
):
    """Violin plots showing spot silhouette distribution and latent feature distributions per domain."""
    pred_labels = np.asarray(predicted_domain).astype(str)
    sample_sils = silhouette_samples(embeddings, pred_labels)

    df_plot = pd.DataFrame({
        'Domain': pred_labels,
        'Silhouette': sample_sils,
        'Latent_Dim1': embeddings[:, 0] if embeddings.shape[1] > 0 else 0
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.set_theme(style="whitegrid")

    sns.violinplot(x='Domain', y='Silhouette', data=df_plot, ax=axes[0], palette="tab20", inner="quartile")
    axes[0].set_title(f"Silhouette Score Distribution per Domain\n{dataset_name}", fontsize=11, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)

    sns.violinplot(x='Domain', y='Latent_Dim1', data=df_plot, ax=axes[1], palette="tab20", inner="quartile")
    axes[1].set_title(f"Latent Dimension 1 Distribution per Domain\n{dataset_name}", fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎻 Violin plots saved to -> {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_all_results(
    results: dict,
    adata_RNA: ad.AnnData,
    dataset_name: str,
    rna_pca_comps: int,
    seed: int,
    plots_dir: str = "results_arise_plus_v2_dec/plots",
    show: bool = False
):
    """Generates and saves the full visualization suite."""
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap_silhouette"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    cell_positions = adata_RNA.obsm['spatial']
    ground_truth = adata_RNA.obs['ground_truth'].values
    best_labels = results['best_labels']
    best_embeddings = results['best_embeddings']
    best_ari = results['best_ari']
    best_sil = results['best_silhouette']

    # 1. Training Curves
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_pca{rna_pca_comps}_training_curves.png")
    plot_training_curves(results, f"{dataset_name} (PCA {rna_pca_comps}, Seed {seed})", save_path=curve_path, show=show)

    # 2. Spatial Domains Map
    spatial_path = os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_pca{rna_pca_comps}_spatial_domains.png")
    plot_spatial_domains(cell_positions, ground_truth, best_labels, dataset_name, best_ari, save_path=spatial_path, show=show)

    # 3. UMAP & Silhouette Report
    umap_path = os.path.join(plots_dir, "umap_silhouette", f"{dataset_name}_seed{seed}_pca{rna_pca_comps}_umap_silhouette.png")
    plot_umap_and_silhouette(best_embeddings, ground_truth, best_labels, dataset_name, best_ari, best_sil, save_path=umap_path, show=show)

    # 4. Violin Distributions
    violin_path = os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_pca{rna_pca_comps}_violin_profiles.png")
    plot_violin_profiles(best_embeddings, best_labels, dataset_name, save_path=violin_path, show=show)


# ==============================================================================
# 12. Benchmark Execution Pipeline
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
        os.system(f'gdown --folder "{meta["url"]}" -O "{base}"')

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


def run_experiment(
    datasets: Union[str, List[Union[str, int]]] = "all",
    seeds: List[int] = DEFAULT_SEEDS,
    rna_pca_comps: int = 60,
    epochs: int = 400,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    out_dim: int = 64,
    visualize: bool = True,
    device: Optional[str] = None,
    base_data_dir: str = "data",
    output_dir: str = "results_arise_plus_v2_dec"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main Runner across datasets and seeds:
      - 4-Stream 1-Layer GCNs
      - RNA PCA Reduction
      - Spatially-Masked Cross-Attention
      - Dense Relational Gram Matrix Alignment
      - Sinkhorn Entropic Optimal Transport Loss
      - Spatial Potts MRF-Regularized Consensus DEC (Two-Stage Training)
      - Kendall & Gal Adaptive Multi-Task Uncertainty Loss
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 1 / Stage 2 epoch breakdown
    if pretrain_epochs is None or finetune_epochs is None:
        pretrain_epochs = int(epochs * 0.625)
        finetune_epochs = epochs - pretrain_epochs

    # Parse target datasets
    if datasets == "all":
        target_datasets = DATASET_LIST
    elif isinstance(datasets, (int, str)):
        if isinstance(datasets, int) or str(datasets).isdigit():
            target_datasets = [DATASET_LIST[int(datasets)]]
        else:
            target_datasets = [datasets]
    else:
        target_datasets = []
        for d in datasets:
            if isinstance(d, int) or str(d).isdigit():
                target_datasets.append(DATASET_LIST[int(d)])
            else:
                target_datasets.append(d)

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    print("\n" + "=" * 80)
    print(" 🚀 STARTING 4-ENCODER 1-LAYER PLUS-V2 + SPATIAL POTTS DEC BENCHMARK")
    print("=" * 80)
    print(f"• Datasets     : {target_datasets}")
    print(f"• Seeds ({len(seeds)})   : {seeds}")
    print(f"• RNA PCA Comps: {rna_pca_comps}")
    print(f"• Total Epochs : {epochs} (Pretrain: {pretrain_epochs}, DEC: {finetune_epochs}) | LR: {lr}")
    print(f"• Loss Protocol: Multi-Task Recon + Spatial Contrastive + Gram Alignment + Sinkhorn OT + Potts DEC")
    print(f"• Loss Weights : Kendall & Gal Adaptive Uncertainty Balancing (5 Tasks)")
    print(f"• Architecture : 4-Stream 1-Layer GCN (Hidden={hidden_dim}, Out={out_dim}) | Device: {device}")
    print("=" * 80 + "\n")

    all_records = []

    for ds_name in target_datasets:
        print("\n" + "#" * 80)
        print(f"####################### DATASET: {ds_name} #######################")
        print("#" * 80)

        # 1. Load Data
        adata_RNA, adata_omics2, cell_positions, num_clusters = load_dataset(ds_name, base_data_dir=base_data_dir)
        true_labels = adata_RNA.obs['ground_truth'].values

        # 2. Preprocess with RNA PCA
        print(f"Preprocessing {ds_name} (RNA PCA components = {rna_pca_comps})...")
        x_rna_pca, x_aux_feat = preprocess_with_rna_pca(
            adata_RNA, adata_omics2, ds_name, rna_pca_comps=rna_pca_comps
        )
        print(f"  • RNA Features: {x_rna_pca.shape} | Aux Features: {x_aux_feat.shape} | Clusters: {num_clusters}")

        for seed in seeds:
            print(f"\n--- Running Seed: {seed} on {ds_name} ---")
            set_seed(seed)

            # 3. Construct 4-Encoder Graphs with Spatial Mask
            data = build_4encoder_plus_graphs(
                x_rna=x_rna_pca,
                x_aux=x_aux_feat,
                cell_positions=cell_positions,
                device=device,
                num_neighbors=15
            )

            # 4. Instantiate Model
            model = Dual4Encoder1LayerPlusV2DEC(
                in_rna_dim=x_rna_pca.shape[1],
                in_aux_dim=x_aux_feat.shape[1],
                num_clusters=num_clusters,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                dropout=0.0
            ).to(device)

            # 5. Two-Stage Training (Pre-train + Spatial Potts DEC)
            start_time = time.time()
            results = train_model_plus_v2_dec(
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
            runtime_sec = time.time() - start_time

            # 6. Evaluate Final Metrics
            best_labels = results['best_labels']
            best_embeddings = results['best_embeddings']
            best_sil = results['best_silhouette']
            best_ep_sil = results['best_epoch_silhouette']
            stage_sil = results['best_stage_silhouette']

            metrics = compute_all_metrics(true_labels, best_labels, embeddings=best_embeddings)
            metrics['Dataset'] = ds_name
            metrics['Seed'] = seed
            metrics['RNA_PCA_Comps'] = rna_pca_comps
            metrics['Best_Epoch_Sil'] = best_ep_sil
            metrics['Best_Stage_Sil'] = stage_sil
            metrics['Peak_ARI'] = results['best_ari']
            metrics['Peak_ARI_Epoch'] = results['best_epoch_ari']
            metrics['Peak_ARI_Stage'] = results['best_stage_ari']
            metrics['Runtime_Sec'] = round(runtime_sec, 2)

            all_records.append(metrics)

            print(f"\n✅ Result [{ds_name} | Seed {seed}]:")
            print(f"   • ARI (Best Sil): {metrics['ARI']:.4f} (at Ep {best_ep_sil}, {stage_sil})")
            print(f"   • Peak ARI      : {metrics['Peak_ARI']:.4f} (at Ep {metrics['Peak_ARI_Epoch']}, {metrics['Peak_ARI_Stage']})")
            print(f"   • NMI: {metrics['NMI']:.4f} | AMI: {metrics['AMI']:.4f} | FMI: {metrics['FMI']:.4f}")
            print(f"   • Silhouette    : {best_sil:.4f} | Runtime: {runtime_sec:.1f}s")

            # 7. Visual Analytics Suite
            if visualize:
                print(f"Generating full diagnostic visualization suite for {ds_name} (Seed {seed})...")
                adata_RNA.obs['predicted_domain'] = best_labels.astype(str)
                visualize_all_results(
                    results=results,
                    adata_RNA=adata_RNA,
                    dataset_name=ds_name,
                    rna_pca_comps=rna_pca_comps,
                    seed=seed,
                    plots_dir=plots_dir,
                    show=False
                )

        # Save per-dataset intermediate CSV
        df_inter = pd.DataFrame([r for r in all_records if r['Dataset'] == ds_name])
        ds_csv_path = os.path.join(output_dir, f"Arise4Encoder1Layer_PlusV2_DEC_{ds_name}.csv")
        df_inter.to_csv(ds_csv_path, index=False)
        print(f"\n📁 Saved intermediate results for {ds_name} -> {ds_csv_path}")

    # ==========================================================================
    # Summary Tables & Aggregation
    # ==========================================================================
    df_all = pd.DataFrame(all_records)
    metric_cols = ['ARI', 'NMI', 'AMI', 'Homo', 'V-Measure', 'FMI', 'Silhouette', 'Peak_ARI', 'Runtime_Sec']

    summary_rows = []
    for ds_name, grp in df_all.groupby('Dataset'):
        row = {'Dataset': ds_name}
        for m in metric_cols:
            mean_val = grp[m].mean()
            std_val = grp[m].std()
            row[f"{m}_Mean"] = mean_val
            row[f"{m}_Std"] = std_val
            row[f"{m}_Summary"] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Save to disk
    full_csv = os.path.join(output_dir, "arise_4encoder_1layer_plus_v2_dec_results.csv")
    agg_csv = os.path.join(output_dir, "arise_4encoder_1layer_plus_v2_dec_aggregated.csv")
    df_all.to_csv(full_csv, index=False)
    df_summary.to_csv(agg_csv, index=False)

    print("\n" + "=" * 80)
    print(" 🏆 FINAL AGGREGATED BENCHMARK SUMMARY (MEAN ± STD)")
    print("=" * 80)
    display_cols = ['Dataset'] + [f"{m}_Summary" for m in ['ARI', 'Peak_ARI', 'NMI', 'Silhouette', 'Runtime_Sec']]
    print(df_summary[display_cols].to_string(index=False))
    print("=" * 80)
    print(f"📊 Detailed run CSV saved to      -> {full_csv}")
    print(f"📈 Aggregated summary saved to   -> {agg_csv}")
    print(f"🎨 Visualizations saved to       -> {plots_dir}/")
    print("=" * 80 + "\n")

    return df_all, df_summary


# ==============================================================================
# 13. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arise 4-Encoder 1-Layer PlusV2 with Spatial Potts DEC Runner")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['all'],
        help="Datasets to evaluate (e.g., 'all', 0 1 2, or '10x_human_lymph_node_A1')"
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help="Random seeds for evaluation (default: 42 1234 2024)"
    )
    parser.add_argument(
        '--rna_pca_comps',
        type=int,
        default=60,
        help="Number of PCA components for RNA (default: 60)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=400,
        help="Total epochs (default: 400 -> 250 pretrain, 150 DEC)"
    )
    parser.add_argument(
        '--pretrain_epochs',
        type=int,
        default=None,
        help="Explicit pretrain epochs (default: 250)"
    )
    parser.add_argument(
        '--finetune_epochs',
        type=int,
        default=None,
        help="Explicit DEC finetune epochs (default: 150)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=512,
        help="Hidden dimension for decoders (default: 512)"
    )
    parser.add_argument(
        '--out_dim',
        type=int,
        default=64,
        help="Embedding / bottleneck dimension (default: 64)"
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help="Generate full diagnostic visualization suite (default: True)"
    )
    parser.add_argument(
        '--no_visualize',
        action='store_false',
        dest='visualize',
        help="Disable visualizations"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="results_arise_plus_v2_dec",
        help="Output directory for CSVs and plots (default: results_arise_plus_v2_dec)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Computation device ('cuda' or 'cpu')"
    )

    args = parser.parse_args()

    parsed_datasets = args.datasets
    if len(parsed_datasets) == 1 and parsed_datasets[0] == 'all':
        parsed_datasets = "all"

    run_experiment(
        datasets=parsed_datasets,
        seeds=args.seeds,
        rna_pca_comps=args.rna_pca_comps,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        visualize=args.visualize,
        device=args.device,
        output_dir=args.output_dir
    )
