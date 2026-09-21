#!/usr/bin/env python3
"""
================================================================================
  🧬 AriseSpatialGlue_4Encoder_1Layer_PlusV2.py

  4-Encoder 1-Layer Arise Architecture with Synergistic Loss Functions:
  1. 📐 4-Stream 1-Layer GCNs (RNA Spatial, RNA Sim, Aux Spatial, Aux Sim)
  2. 🧬 RNA Dimensionality Reduction via PCA (n_comps = 60 / 100)
  3. ⚡ Spatially-Constrained Cross-Attention Feature Exchange (Graph-Masked)
  4. 🔗 Dense Relational Gram Matrix Alignment (Frobenius Norm)
  5. 🌊 Entropic Sinkhorn Optimal Transport Relational Loss
  6. ⚖️ Kendall & Gal Uncertainty Multi-Task Adaptive Loss Weighting (4 Objectives)
  7. 📈 Complete Diagnostic Visual Analytics Suite:
     - Multi-Task Training Curves (Total Loss, Silhouette, ARI)
     - Spatial Domain Clustering Maps (Ground Truth vs. Model Predictions)
     - 4-Panel UMAP & Silhouette Diagnostic Reports (Latent Space + Knife Plot)
     - Violin Profiles: Spot Silhouette Profiles & Latent Dimension Distributions
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
        sc.pp.highly_variable_genes(adata_RNA, flavor="seurat", n_top_genes=3000)
        sc.pp.scale(adata_RNA)

    adata_RNA_high = adata_RNA[:, adata_RNA.var['highly_variable']].copy()
    actual_pca_comps = min(rna_pca_comps, adata_RNA_high.n_vars, adata_RNA_high.n_obs - 1)
    RNA_expression = pca(adata_RNA_high, n_comps=actual_pca_comps)

    adata_omics2_copy = adata_omics2[adata_RNA.obs_names].copy()
    if dataset_name.startswith("10x"):
        adata_omics2_copy = clr_normalize_each_cell(adata_omics2_copy)
        sc.pp.scale(adata_omics2_copy)
        ADT_expression = adata_omics2_copy.X
    else:
        adata_omics2_copy.X = tfidf(adata_omics2_copy.X)
        sc.pp.normalize_per_cell(adata_omics2_copy, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2_copy)
        n_comps = min(60, adata_omics2_copy.shape[1])
        adata_omics2_copy.obsm['feat'] = pca(adata_omics2_copy, n_comps=n_comps)
        ADT_expression = adata_omics2_copy.obsm['feat']

    if sp.issparse(ADT_expression):
        ADT_expression = ADT_expression.toarray()

    return RNA_expression, ADT_expression


# ==============================================================================
# 4. 4-Encoder Graph Construction (Spatial & Cosine Similarity)
# ==============================================================================

class Dual4GraphPlusData(Data):
    """PyG Data container holding 4 graphs and spatial neighborhood masks."""
    def __init__(self, x_RNA, x_ADT,
                 sim_edge_index_rna, sim_edge_weight_rna,
                 dist_edge_index_rna, dist_edge_weight_rna,
                 sim_edge_index_aux, sim_edge_weight_aux,
                 dist_edge_index_aux, dist_edge_weight_aux,
                 spatial_mask, spatial_adj):
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
        self.spatial_mask = spatial_mask
        self.spatial_adj = spatial_adj


def build_4encoder_graphs(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15) -> Dual4GraphPlusData:
    """
    Constructs 4 distinct graph streams:
      1. Spatial KNN Distance Graph for RNA
      2. Cosine Similarity kNN Graph for RNA
      3. Spatial KNN Distance Graph for Aux
      4. Cosine Similarity kNN Graph for Aux
    """
    num_nodes = RNA_expression.shape[0]

    # 1. Spatial Euclidean KNN Graph
    knn_spatial = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_spatial = knn_spatial.maximum(knn_spatial.T)

    dist_edge_index = torch.tensor(np.array(knn_spatial.nonzero()), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_spatial.data, dtype=torch.float).to(device)

    # 2. RNA Cosine Similarity Graph
    sim_matrix_rna = cosine_similarity(RNA_expression)
    nbrs_rna = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(RNA_expression)
    _, indices_rna = nbrs_rna.kneighbors(RNA_expression)

    adj_rna = np.zeros_like(sim_matrix_rna, dtype=int)
    for i in range(num_nodes):
        for j in indices_rna[i][1:]:
            adj_rna[i, j] = 1
            adj_rna[j, i] = 1

    sim_edge_index_rna = torch.tensor(np.array(np.nonzero(adj_rna)), dtype=torch.long).to(device)
    sim_edge_weight_rna = torch.tensor(sim_matrix_rna[adj_rna > 0], dtype=torch.float).to(device)

    # 3. Aux Cosine Similarity Graph
    sim_matrix_aux = cosine_similarity(ADT_expression)
    nbrs_aux = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(ADT_expression)
    _, indices_aux = nbrs_aux.kneighbors(ADT_expression)

    adj_aux = np.zeros_like(sim_matrix_aux, dtype=int)
    for i in range(num_nodes):
        for j in indices_aux[i][1:]:
            adj_aux[i, j] = 1
            adj_aux[j, i] = 1

    sim_edge_index_aux = torch.tensor(np.array(np.nonzero(adj_aux)), dtype=torch.long).to(device)
    sim_edge_weight_aux = torch.tensor(sim_matrix_aux[adj_aux > 0], dtype=torch.float).to(device)

    # Dense Spatial Adjacency Matrix & Mask for Cross-Attention (V11)
    spatial_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    spatial_adj[dist_edge_index[0], dist_edge_index[1]] = 1.0
    spatial_mask = spatial_adj.clone()
    spatial_mask.fill_diagonal_(1.0)

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return Dual4GraphPlusData(
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
        spatial_mask=spatial_mask,
        spatial_adj=spatial_adj
    )


# ==============================================================================
# 5. Spatially-Constrained Cross-Attention (V11)
# ==============================================================================

class GraphMaskedCrossAttention(nn.Module):
    """
    Spatially-constrained cross-modal multi-head attention where RNA queries
    attend exclusively over 1-hop spatial graph neighbor keys in the auxiliary modality.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

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
# 6. Entropic Sinkhorn Optimal Transport Relational Loss (V12)
# ==============================================================================

class SinkhornOptimalTransportLoss(nn.Module):
    """
    Differentiable Entropic Optimal Transport (Sinkhorn-Wasserstein distance)
    aligning the probability distributions across RNA and Auxiliary modalities.
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
# 7. Model Architecture: 4-Encoder 1-Layer GCN + Synergistic Loss
# ==============================================================================

class Dual4Encoder1LayerPlusV2(nn.Module):
    """
    🧬 4-Encoder 1-Layer Architecture with Synergistic Multi-Task Loss:
      - 4 Streams of 1-Layer GCNs (RNA Spatial, RNA Sim, Aux Spatial, Aux Sim)
      - Spatially-Constrained Cross-Attention (V11)
      - Intra-omic and Inter-omic Linear Fusion
      - Multi-Head Reconstruction Decoders
      - Dense Relational Gram Matrix Alignment (V7)
      - Sinkhorn Entropic Optimal Transport Loss (V12)
      - Kendall & Gal Adaptive Uncertainty Multi-Task Loss Balancing
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

        # --- Kendall & Gal Homoscedastic Multi-Task Uncertainty Parameters (s_0..s_3) ---
        # Task losses: [L_recon, L_spatial, L_dense, L_ot]
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, data: Dual4GraphPlusData) -> Dict[str, torch.Tensor]:
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

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'aux_s': aux_s,
            'aux_d': aux_d,
            'fused_rna': z_rna_refined,
            'fused_aux': fused_aux,
            'fused_joint': fused_joint,
            'embedding': fused_joint
        }

    def compute_loss(self, data: Dual4GraphPlusData, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
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

        total_loss = (0.5 * p0 * total_recon + 0.5 * self.log_vars[0] +
                      0.5 * p1 * l_spatial + 0.5 * self.log_vars[1] +
                      0.5 * p2 * l_dense + 0.5 * self.log_vars[2] +
                      0.5 * p3 * l_ot + 0.5 * self.log_vars[3])

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_recon': total_recon.item(),
            'loss_spatial': l_spatial.item(),
            'loss_dense': l_dense.item(),
            'loss_ot': l_ot.item()
        }

        return total_loss, loss_dict


# ==============================================================================
# 8. Training & Evaluation Engine
# ==============================================================================

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int, random_state: int = 42) -> np.ndarray:
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return kmeans.fit_predict(embeddings)


def train_model_plus_v2(
    model: Dual4Encoder1LayerPlusV2,
    data: Dual4GraphPlusData,
    epochs: int = 400,
    lr: float = 1e-3,
    num_clusters: int = 10,
    true_labels: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Union[float, int, np.ndarray, List[float]]]:
    """
    End-to-End Training Loop with Kendall & Gal Adaptive Multi-Task Uncertainty Loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_sil = -1.0
    best_epoch_sil = 0
    best_ari_val = -1.0
    best_epoch_ari = 0
    best_embeddings = None
    best_labels = None

    loss_history = []
    loss_recon_history = []
    loss_spatial_history = []
    loss_dense_history = []
    loss_ot_history = []
    epoch_sil_history = []
    epoch_ari_history = []

    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss, loss_dict = model.compute_loss(data, outputs)
        loss.backward()
        optimizer.step()

        loss_history.append(loss_dict['loss_total'])
        loss_recon_history.append(loss_dict['loss_recon'])
        loss_spatial_history.append(loss_dict['loss_spatial'])
        loss_dense_history.append(loss_dict['loss_dense'])
        loss_ot_history.append(loss_dict['loss_ot'])

        # Evaluate representation
        model.eval()
        with torch.no_grad():
            eval_out = model(data)
            emb = eval_out['embedding'].detach().cpu().numpy()
        model.train()

        pred_labels = cluster_embeddings(emb, num_clusters, random_state=seed)
        sil = float(silhouette_score(emb, pred_labels))
        epoch_sil_history.append(sil)

        if true_labels is not None:
            ari = float(adjusted_rand_score(true_labels, pred_labels))
            epoch_ari_history.append(ari)
        else:
            ari = None

        curr_epoch = epoch + 1
        if sil > best_sil:
            best_sil = sil
            best_epoch_sil = curr_epoch
            best_embeddings = emb.copy()
            best_labels = pred_labels.copy()

        if ari is not None and ari > best_ari_val:
            best_ari_val = ari
            best_epoch_ari = curr_epoch

        postfix = {'loss': f"{loss.item():.3f}", 'sil': f"{sil:.4f}", 'ot': f"{loss_dict['loss_ot']:.3f}"}
        if ari is not None:
            postfix['ari'] = f"{ari:.4f}"
        postfix['best_sil'] = f"{best_sil:.4f} (ep {best_epoch_sil})"
        pbar.set_postfix(postfix)

        if verbose and (curr_epoch % 50 == 0 or curr_epoch == 1 or curr_epoch == epochs):
            ari_str = f" | ARI: {ari:.4f}" if ari is not None else ""
            tqdm.write(f"Epoch {curr_epoch:3d}/{epochs} | Total Loss: {loss.item():.4f} | Recon: {loss_dict['loss_recon']:.4f} | OT: {loss_dict['loss_ot']:.4f} | Sil: {sil:.4f}{ari_str} | Best Sil: {best_sil:.4f} (Epoch {best_epoch_sil})")

    last_sil = epoch_sil_history[-1] if epoch_sil_history else 0.0
    last_ari = epoch_ari_history[-1] if epoch_ari_history else 0.0

    if true_labels is not None and best_labels is not None:
        final_ari = float(adjusted_rand_score(true_labels, best_labels))
        final_nmi = float(normalized_mutual_info_score(true_labels, best_labels))
    else:
        final_ari, final_nmi = 0.0, 0.0

    if verbose:
        print("\n" + "=" * 70)
        print(" 🧬 TRAINING FINISHED - SUMMARY ".center(70, "="))
        print("=" * 70)
        print(f"Final Total Loss               : {loss.item():.4f}")
        print(f"Last Epoch Silhouette Score    : {last_sil:.4f}")
        if true_labels is not None:
            print(f"Last Epoch ARI                 : {last_ari:.4f}")
        print("-" * 70)
        print(f"Best Silhouette Score          : {best_sil:.4f} (Achieved at Epoch {best_epoch_sil})")
        if true_labels is not None:
            print(f"ARI at Best Silhouette Epoch   : {final_ari:.4f}")
            print(f"NMI at Best Silhouette Epoch   : {final_nmi:.4f}")
            print(f"Peak ARI Achieved Across Run   : {best_ari_val:.4f} (Achieved at Epoch {best_epoch_ari})")
        print("=" * 70 + "\n")

    return {
        'model': model,
        'best_embeddings': best_embeddings,
        'best_labels': best_labels,
        'best_sil': best_sil,
        'best_epoch_sil': best_epoch_sil,
        'best_ari': final_ari,
        'best_nmi': final_nmi,
        'peak_ari': best_ari_val,
        'best_epoch_ari': best_epoch_ari,
        'last_sil': last_sil,
        'last_ari': last_ari,
        'loss_history': loss_history,
        'loss_recon_history': loss_recon_history,
        'loss_spatial_history': loss_spatial_history,
        'loss_dense_history': loss_dense_history,
        'loss_ot_history': loss_ot_history,
        'epoch_sil_history': epoch_sil_history,
        'epoch_ari_history': epoch_ari_history
    }


# ==============================================================================
# 9. Comprehensive Visual Analytics Suite
# ==============================================================================

def plot_training_curves(training_results: Dict, dataset_name: str = "Dataset", save_path: Optional[str] = None, show: bool = True):
    """
    Plots a 3-panel diagnostic training curve figure:
      1. Total Loss Curve
      2. Silhouette Score Curve across Epochs (with Best Epoch marker)
      3. ARI Curve across Epochs (with Best Silhouette Epoch & Peak ARI marker)
    """
    total_epochs = len(training_results['loss_history'])
    epochs_range = range(1, total_epochs + 1)
    has_ari = len(training_results.get('epoch_ari_history', [])) > 0

    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Total Loss Curve
    axes[0].plot(epochs_range, training_results['loss_history'], color='#1f77b4', linewidth=2.0, label='Total Loss')
    axes[0].set_title(f'Multi-Task Loss Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Multi-Task Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True, fontsize=9)

    # 2. Silhouette Score Curve
    axes[1].plot(epochs_range, training_results['epoch_sil_history'], color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    best_ep = training_results.get('best_epoch_sil', 0)
    best_sil = training_results.get('best_sil', 0.0)
    if best_ep > 0:
        axes[1].axvline(x=best_ep, color='#d62728', linestyle='--', linewidth=1.5, label=f'Best Sil @ Ep {best_ep} ({best_sil:.4f})')
        axes[1].scatter([best_ep], [best_sil], color='#d62728', s=60, zorder=5)
    axes[1].set_title(f'Silhouette Score Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc='lower right', fontsize=9)

    # 3. ARI Curve
    if has_ari:
        axes[2].plot(epochs_range, training_results['epoch_ari_history'], color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        if best_ep > 0 and len(training_results['epoch_ari_history']) >= best_ep:
            ari_at_best = training_results['epoch_ari_history'][best_ep - 1]
            axes[2].axvline(x=best_ep, color='#d62728', linestyle='--', linewidth=1.5, label=f'Best Sil Ep {best_ep} (ARI: {ari_at_best:.4f})')
            axes[2].scatter([best_ep], [ari_at_best], color='#d62728', s=60, zorder=5)
        peak_ari = training_results.get('peak_ari', 0.0)
        peak_ep = training_results.get('best_epoch_ari', 0)
        if peak_ep > 0 and peak_ari > 0:
            axes[2].scatter([peak_ep], [peak_ari], color='#9467bd', marker='*', s=100, zorder=6, label=f'Peak ARI @ Ep {peak_ep} ({peak_ari:.4f})')

        axes[2].set_title(f'ARI Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(frameon=True, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved diagnostic training curves to: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_spatial_domains(
    cell_positions: np.ndarray,
    ground_truth: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str,
    ari: float,
    out_path: str,
    seed: int = 42,
    show: bool = False
):
    """Side-by-side spatial coordinates comparison of Ground Truth and Predicted Domains."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)

    gt_cats = pd.Categorical(ground_truth)
    pred_cats = pd.Categorical(pred_labels)

    axes[0].scatter(cell_positions[:, 0], cell_positions[:, 1], c=gt_cats.codes, cmap='tab20', s=12, alpha=0.9)
    axes[0].set_title(f"Ground Truth Annotations ({dataset_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Spatial X", fontsize=10)
    axes[0].set_ylabel("Spatial Y", fontsize=10)
    axes[0].grid(False)

    axes[1].scatter(cell_positions[:, 0], cell_positions[:, 1], c=pred_cats.codes, cmap='tab20', s=12, alpha=0.9)
    axes[1].set_title(f"Arise 4-Encoder Predicted Domains (ARI: {ari:.4f})", fontsize=12, fontweight='bold', color='#1d4ed8')
    axes[1].set_xlabel("Spatial X", fontsize=10)
    axes[1].set_ylabel("Spatial Y", fontsize=10)
    axes[1].grid(False)

    plt.suptitle(f"Spatial Domains Comparison - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Saved spatial domains map to: {out_path}")


def compute_umap_projection(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.3, random_state: int = 42) -> np.ndarray:
    """Computes a 2D UMAP projection from high-dimensional latent embeddings."""
    try:
        temp_adata = ad.AnnData(X=embeddings.astype(np.float32))
        sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X', random_state=random_state)
        sc.tl.umap(temp_adata, min_dist=min_dist, random_state=random_state)
        return temp_adata.obsm['X_umap']
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=random_state).fit_transform(embeddings)


def plot_umap_and_silhouette(
    embeddings: np.ndarray,
    ground_truth: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str,
    metrics: Dict[str, float],
    out_path: str,
    seed: int = 42,
    show: bool = False
):
    """
    Generates a 4-panel diagnostic figure:
      1. 2D UMAP colored by Ground Truth annotations
      2. 2D UMAP colored by Predicted Domains
      3. 2D UMAP colored continuously by Cell-Level Silhouette Coefficients
      4. Per-Cluster Silhouette Profile (Knife Plot)
    """
    umap_coords = compute_umap_projection(embeddings, n_neighbors=15, min_dist=0.3, random_state=seed)
    sample_sil_values = silhouette_samples(embeddings, pred_labels)
    avg_sil = metrics.get('Silhouette', float(silhouette_score(embeddings, pred_labels)))
    ari = metrics.get('ARI', 0.0)
    nmi = metrics.get('NMI', 0.0)

    unique_preds = np.unique(pred_labels)
    n_clusters = len(unique_preds)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=150)

    # 1. UMAP Ground Truth
    gt_labels_str = [f"Class {g}" if isinstance(g, (int, np.integer)) else str(g) for g in ground_truth]
    df_umap = pd.DataFrame({
        'UMAP1': umap_coords[:, 0],
        'UMAP2': umap_coords[:, 1],
        'GT': gt_labels_str,
        'Pred': [f"Domain {p}" for p in pred_labels],
        'Sil': sample_sil_values
    })

    sns.scatterplot(
        data=df_umap, x='UMAP1', y='UMAP2', hue='GT', palette='tab20',
        s=25, ax=axes[0, 0], alpha=0.85, edgecolor='none'
    )
    axes[0, 0].set_title(f"UMAP: Ground Truth Annotations ({dataset_name})", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("UMAP 1", fontsize=10)
    axes[0, 0].set_ylabel("UMAP 2", fontsize=10)
    axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)

    # 2. UMAP Predicted Domains
    sns.scatterplot(
        data=df_umap, x='UMAP1', y='UMAP2', hue='Pred', palette='tab20',
        s=25, ax=axes[0, 1], alpha=0.85, edgecolor='none'
    )
    axes[0, 1].set_title(f"UMAP: Predicted Domains (ARI: {ari:.4f} | NMI: {nmi:.4f})", fontsize=12, fontweight='bold', color='#1d4ed8')
    axes[0, 1].set_xlabel("UMAP 1", fontsize=10)
    axes[0, 1].set_ylabel("UMAP 2", fontsize=10)
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)

    # 3. UMAP Cell-Level Silhouette Heatmap
    vmin = min(0.0, float(sample_sil_values.min()))
    vmax = max(0.4, float(sample_sil_values.max()))
    sc_plot = axes[1, 0].scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=sample_sil_values, cmap='coolwarm', vmin=vmin, vmax=vmax,
        s=25, alpha=0.85, edgecolor='none'
    )
    axes[1, 0].set_title(f"UMAP: Cell Silhouette Quality (Mean: {avg_sil:.4f})", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("UMAP 1", fontsize=10)
    axes[1, 0].set_ylabel("UMAP 2", fontsize=10)
    cbar = plt.colorbar(sc_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label('Silhouette Coefficient', fontsize=10)

    # 4. Silhouette Profile (Knife Plot)
    y_lower = 10
    palette = sns.color_palette('tab20', n_clusters)
    for idx, c_label in enumerate(unique_preds):
        c_sil = sample_sil_values[pred_labels == c_label]
        c_sil.sort()
        size_c = c_sil.shape[0]
        y_upper = y_lower + size_c
        color = palette[idx % len(palette)]
        axes[1, 1].fill_betweenx(
            np.arange(y_lower, y_upper), 0, c_sil,
            facecolor=color, edgecolor=color, alpha=0.75, label=f"Domain {c_label}"
        )
        axes[1, 1].text(-0.05, y_lower + 0.5 * size_c, str(c_label), fontsize=9, fontweight='bold', va='center')
        y_lower = y_upper + 10

    axes[1, 1].axvline(x=avg_sil, color='red', linestyle='--', linewidth=1.8, label=f"Mean Sil ({avg_sil:.4f})")
    axes[1, 1].set_title(f"Per-Domain Silhouette Profile (Overall Sil: {avg_sil:.4f})", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Silhouette Coefficient Values", fontsize=10)
    axes[1, 1].set_ylabel("Domain Clusters / Cells", fontsize=10)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlim([min(-0.2, sample_sil_values.min() - 0.05), 1.0])
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)

    plt.suptitle(f"UMAP & Silhouette Quality Report - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Saved UMAP & Silhouette diagnostic report to: {out_path}")


def plot_violin_profiles(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
    dataset_name: str,
    sil: float,
    out_path: str,
    seed: int = 42,
    show: bool = False
):
    """Spot-level silhouette distribution and Latent Dimension 1 violin profiles."""
    sample_sil_values = silhouette_samples(embeddings, pred_labels)

    df_violin = pd.DataFrame({
        'predicted_domain': [f"Domain {l}" for l in pred_labels],
        'silhouette_coefficient': sample_sil_values,
        'Latent_Dim_1': embeddings[:, 0],
        'Latent_Dim_2': embeddings[:, 1] if embeddings.shape[1] > 1 else embeddings[:, 0]
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150)

    # 1. Silhouette Coefficient Violin Plot
    sns.violinplot(
        data=df_violin,
        x='predicted_domain',
        y='silhouette_coefficient',
        palette='Set2',
        inner='quartile',
        ax=axes[0]
    )
    axes[0].axhline(sil, color='red', linestyle='--', label=f'Mean Sil: {sil:.4f}')
    axes[0].set_title("Silhouette Coefficient per Predicted Domain", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Domain", fontsize=10)
    axes[0].set_ylabel("Silhouette Coefficient", fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=30)

    # 2. Latent Dimension 1 Violin Plot
    sns.violinplot(
        data=df_violin,
        x='predicted_domain',
        y='Latent_Dim_1',
        palette='tab10',
        inner='box',
        ax=axes[1]
    )
    axes[1].set_title("Latent Dimension 1 Distribution per Domain", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Predicted Domain", fontsize=10)
    axes[1].set_ylabel("Latent Embedding Dim 1", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=30)

    plt.suptitle(f"Violin Clustering Profiles - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    print(f"Saved violin clustering profiles to: {out_path}")


def plot_all_visualizations_plus_v2(
    adata_RNA: ad.AnnData,
    results: Dict,
    cell_positions: np.ndarray,
    ground_truth: np.ndarray,
    dataset_name: str = "Dataset",
    seed: int = 42,
    rna_pca_comps: int = 60,
    output_dir: str = "results_arise_plus_v2",
    show: bool = False
):
    """Generate and save all 4 publication-ready visualization modules."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap_silhouette"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    best_embeddings = results['best_embeddings']
    best_labels = results['best_labels']
    sil = results['best_sil']
    ari = results.get('best_ari', 0.0)

    # 1. 📈 Diagnostic Multi-Task Training Curves
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_training_curves.png")
    plot_training_curves(
        results,
        dataset_name=f"{dataset_name} (PCA {rna_pca_comps}, Seed {seed})",
        save_path=curve_path,
        show=show
    )

    # 2. 🗺️ Spatial Domains Comparison Map
    spatial_path = os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_spatial.png")
    plot_spatial_domains(
        cell_positions=cell_positions,
        ground_truth=ground_truth,
        pred_labels=best_labels,
        dataset_name=dataset_name,
        ari=ari,
        out_path=spatial_path,
        seed=seed,
        show=show
    )

    # 3. 🎨 4-Panel UMAP & Silhouette Diagnostic Report
    umap_path = os.path.join(plots_dir, "umap_silhouette", f"{dataset_name}_seed{seed}_umap_silhouette_report.png")
    metrics_dict = {'ARI': ari, 'NMI': results.get('best_nmi', 0.0), 'Silhouette': sil}
    plot_umap_and_silhouette(
        embeddings=best_embeddings,
        ground_truth=ground_truth,
        pred_labels=best_labels,
        dataset_name=dataset_name,
        metrics=metrics_dict,
        out_path=umap_path,
        seed=seed,
        show=show
    )

    # 4. 🎻 Violin Profiles
    violin_path = os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_violin.png")
    plot_violin_profiles(
        embeddings=best_embeddings,
        pred_labels=best_labels,
        dataset_name=dataset_name,
        sil=sil,
        out_path=violin_path,
        seed=seed,
        show=show
    )

    print(f"✅ All visualizations saved into: {plots_dir}/")


# ==============================================================================
# 10. Experiment Runner (Python API)
# ==============================================================================

def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    n_seeds: Optional[int] = None,
    rna_pca_comps: int = 60,
    epochs: int = 400,
    lr: float = 1e-3,
    hidden_dim: int = 512,
    out_dim: int = 64,
    dropout: float = 0.0,
    device: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    output_dir: str = "results_arise_plus_v2",
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Run 4-Encoder 1-Layer Arise with Synergistic Multi-Task Losses across datasets and seeds.
    """
    if device is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device
    dev = torch.device(device_str)

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

    if n_seeds is not None and n_seeds > 0:
        run_seeds = run_seeds[:n_seeds]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "=" * 80)
    print(" 🚀 STARTING 4-ENCODER 1-LAYER (RNA PCA) ARISE-PLUS-V2 EXPERIMENT ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets      : {dataset_names}")
    print(f"• Seeds ({len(run_seeds)})    : {run_seeds}")
    print(f"• RNA PCA Comps : {rna_pca_comps}")
    print(f"• Epochs        : {epochs} | LR: {lr}")
    print(f"• Loss Protocol : Multi-Task Recon + Spatial Contrastive + Gram Alignment + Sinkhorn OT")
    print(f"• Loss Weights  : Kendall & Gal Adaptive Uncertainty Engine (Zero Manual Tuning)")
    print(f"• Dimensions    : Hidden={hidden_dim}, Out={out_dim} | Device: {device_str}")
    print(f"• Output Dir    : {output_dir}")
    print("=" * 80 + "\n")

    for dataset_name in dataset_names:
        cfg = DATASET_REGISTRY[dataset_name]
        print("\n" + "#" * 80)
        print(f" DATASET: {dataset_name} ".center(80, "#"))
        print("#" * 80)

        base = os.path.join(data_dir, dataset_name)
        os.makedirs(base, exist_ok=True)
        rna_path = os.path.join(base, "adata_RNA.h5ad")
        other_path = os.path.join(base, cfg["other_file"])
        annotation_path = os.path.join(base, cfg["anno_file"])

        if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
            print(f"Downloading dataset files into: {base}")
            gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
            os.system(f'{gdown_cmd} --folder "{cfg["url"]}" --output "{base}"')

        # Load AnnData
        adata_RNA = sc.read_h5ad(rna_path)
        adata_omics2 = sc.read_h5ad(other_path)
        adata_RNA.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        anno_df = pd.read_csv(annotation_path, index_col=0)
        true_labels = anno_df[cfg["gt_col"]].values
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
        num_clusters = len(np.unique(true_labels))
        print(f"Dataset: {dataset_name} | Spots: {adata_RNA.n_obs} | Clusters: {num_clusters}")

        # Preprocess with RNA PCA
        RNA_expr, ADT_expr = preprocess_with_rna_pca(
            adata_RNA, adata_omics2, dataset_name, rna_pca_comps=rna_pca_comps
        )
        cell_positions = adata_RNA.obsm['spatial']

        # Construct 4-Encoder Graphs
        data = build_4encoder_graphs(RNA_expr, ADT_expr, cell_positions, device=dev)
        input_dim = data.x_RNA.shape[1]
        adt_dim = data.x_ADT.shape[1]

        dataset_results = []

        for seed in run_seeds:
            print(f"\n" + "-" * 60)
            print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
            print("-" * 60)

            set_seed(seed)

            # Initialize 1-Layer 4-Encoder Model with Synergistic Losses
            model = Dual4Encoder1LayerPlusV2(
                in_rna_dim=input_dim,
                in_aux_dim=adt_dim,
                num_clusters=num_clusters,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                dropout=dropout
            ).to(dev)

            # Train Model
            start_t = time.time()
            result = train_model_plus_v2(
                model=model,
                data=data,
                epochs=epochs,
                lr=lr,
                num_clusters=num_clusters,
                true_labels=true_labels,
                seed=seed,
                verbose=True
            )
            elapsed_time = round(time.time() - start_t, 2)

            best_embeddings = result['best_embeddings']
            best_labels = result['best_labels']
            best_sil = result['best_sil']
            best_epoch_sil = result['best_epoch_sil']
            peak_ari = result['peak_ari']
            best_epoch_ari = result['best_epoch_ari']

            # Compute Final Metrics
            ari = float(adjusted_rand_score(true_labels, best_labels))
            nmi = float(normalized_mutual_info_score(true_labels, best_labels))
            ami = float(adjusted_mutual_info_score(true_labels, best_labels))
            chi = float(calinski_harabasz_score(best_embeddings, best_labels))
            dbi = float(davies_bouldin_score(best_embeddings, best_labels))

            last_sil = result['last_sil']
            last_ari = result['last_ari']

            print(f"Summary for {dataset_name} | Seed: {seed}")
            print(f"• Best Silhouette : {best_sil:.4f} (Epoch {best_epoch_sil}) | ARI @ Best: {ari:.4f} | NMI: {nmi:.4f}")
            print(f"• Peak ARI Run    : {peak_ari:.4f} (Epoch {best_epoch_ari})")
            print(f"• Last Silhouette : {last_sil:.4f} (Epoch {epochs}) | Last ARI: {last_ari:.4f} | Time: {elapsed_time}s")

            # Plot and save all visualizations if requested
            if visualize:
                plot_all_visualizations_plus_v2(
                    adata_RNA=adata_RNA,
                    results=result,
                    cell_positions=cell_positions,
                    ground_truth=true_labels,
                    dataset_name=dataset_name,
                    seed=seed,
                    rna_pca_comps=rna_pca_comps,
                    output_dir=output_dir,
                    show=show_plots
                )

            res_dict = {
                'dataset': dataset_name,
                'seed': seed,
                'Best_Silhouette': best_sil,
                'Best_Sil_Epoch': best_epoch_sil,
                'Best_ARI': ari,
                'Best_NMI': nmi,
                'Peak_ARI': peak_ari,
                'Peak_ARI_Epoch': best_epoch_ari,
                'Last_Silhouette': last_sil,
                'Last_ARI': last_ari,
                'AMI': ami,
                'CHI': chi,
                'DBI': dbi,
                'RNA_PCA_Comps': rna_pca_comps,
                'Runtime_Sec': elapsed_time,
                'no_cluster': num_clusters
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)

        # Save per-dataset results
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"Arise4Encoder1Layer_PlusV2_{dataset_name}_results.csv"), index=False)

    # Save overall summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "Arise4Encoder1Layer_PlusV2_all_results.csv"), index=False)

    summary_cols = ['Best_ARI', 'Peak_ARI', 'Best_Silhouette', 'Last_ARI', 'Last_Silhouette', 'Best_NMI', 'Runtime_Sec']
    print("\n" + "=" * 94)
    print(" ALL EXPERIMENTS COMPLETED - DATASET MEAN SUMMARY ".center(94, "="))
    print("=" * 94)
    print(df_all.groupby('dataset')[summary_cols].mean().to_string())
    print("-" * 94)
    print(" OVERALL MEAN ACROSS ALL DATASETS & SEEDS ".center(94, "-"))
    print(df_all[summary_cols].mean().to_frame().T.to_string(index=False))
    print("=" * 94 + "\n")

    return df_all


# ==============================================================================
# 11. Command-Line Interface (CLI)
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="4-Encoder 1-Layer Arise with Synergistic Multi-Task Losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--datasets', nargs='+', default=['all'],
        help="Datasets to run. Can be index (0 to 5), name (e.g. 10x_human_lymph_node_A1), or 'all'."
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[42, 1234, 2024],
        help="List of random seeds to evaluate (e.g. --seeds 42 1234 2024)."
    )
    parser.add_argument(
        '--n_seeds', type=int, default=None,
        help="Use first N seeds from seed list."
    )
    parser.add_argument(
        '--rna_pca_comps', type=int, default=60,
        help="Number of PCA components for RNA modality (e.g. 60 or 100)."
    )
    parser.add_argument('--epochs', type=int, default=400, help="Total training epochs (e.g. 400 or 350)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--out_dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--device', type=str, default=None, help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--visualize', action='store_true', default=True, help="Generate and save all plots (Curves, Spatial, UMAP, Knife, Violin)")
    parser.add_argument('--show_plots', action='store_true', default=False, help="Display plots interactively")
    parser.add_argument('--output_dir', type=str, default='results_arise_plus_v2', help="Directory to save CSV results and plots")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save/load datasets")

    cli_args = parser.parse_args()

    run_experiment(
        datasets=cli_args.datasets if len(cli_args.datasets) > 1 or cli_args.datasets[0] != 'all' else 'all',
        seeds=cli_args.seeds,
        n_seeds=cli_args.n_seeds,
        rna_pca_comps=cli_args.rna_pca_comps,
        epochs=cli_args.epochs,
        lr=cli_args.lr,
        hidden_dim=cli_args.hidden_dim,
        out_dim=cli_args.out_dim,
        dropout=cli_args.dropout,
        device=cli_args.device,
        visualize=cli_args.visualize,
        show_plots=cli_args.show_plots,
        output_dir=cli_args.output_dir,
        data_dir=cli_args.data_dir
    )
