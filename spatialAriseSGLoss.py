# ==============================================================================
# spatialAriseSGLoss.py
#
# 4-Encoder Arise Architecture with RNA PCA & 1-Layer Encoders:
# - Preprocessing: RNA expression is reduced via PCA to n_comps (e.g. 60 / 100)
# - Encoders: ALL 4 streams use 1-layer GCNs (in_dim -> out_dim):
#     * Encoder 1 (RNA Spatial): 1-layer GCN (dim_rna_pca -> out_dim)
#     * Encoder 2 (RNA Similarity): 1-layer GCN (dim_rna_pca -> out_dim)
#     * Encoder 3 (Aux Spatial): 1-layer GCN (dim_aux -> out_dim)
#     * Encoder 4 (Aux Similarity): 1-layer GCN (dim_aux -> out_dim)
# - Fusion: Arise Linear & MLP concatenation-based fusion
# - Decoders: Arise multi-target reconstruction (RNA PCA, Aux, Joint)
# - Loss: SpatialGlue-Style (Reconstruction + Cycle-Consistency Correspondence Loss)
# - Per-Epoch Logging: Silhouette Score, ARI, and Best Silhouette Epoch reporting
# - Custom Input Support: Datasets, Seeds, Epochs, RNA PCA Comps via CLI or Python API
# ==============================================================================

import os
import sys
import copy
import random
import argparse
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from typing import Optional, List, Union
from tqdm import tqdm

import scanpy as sc
import anndata
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
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
import matplotlib.pyplot as plt
import seaborn as sns


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

DEFAULT_SEEDS = [
    13, 2560, 641, 1892, 1173, 69, 2024, 231, 1971, 2497,
    338, 3127, 2001, 2022, 574, 2428, 999, 1187, 42, 3999
]


# ==============================================================================
# 2. Reproducibility
# ==============================================================================

def set_seed(seed=2024):
    """Set random seed across Python, NumPy, PyTorch, and CUDA."""
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
# 3. Preprocessing (with RNA PCA)
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
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def pca(adata, use_reps=None, n_comps=60):
    """PCA dimensionality reduction."""
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X)
    return feat_pca


def Protein(adata):
    """Normalize ADT (Protein) data using CLR and scale it."""
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    return adata


def tfidf(X):
    """Compute TF-IDF matrix in CSR sparse format."""
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def preprocess_with_rna_pca(adata_RNA, adata_omics2, dataset_name, rna_pca_comps=60):
    """
    Preprocessing with RNA PCA dimensionality reduction:
      - RNA: Normalized, log-transformed, HVG filtered, then reduced via PCA (e.g. 60 or 100 comps).
      - ADT/ATAC: CLR / TF-IDF + PCA.
    """
    # 1. Preprocess RNA
    sc.pp.filter_genes(adata_RNA, min_cells=10)
    sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.scale(adata_RNA)

    adata_RNA_high = adata_RNA[:, adata_RNA.var['highly_variable']].copy()
    actual_pca_comps = min(rna_pca_comps, adata_RNA_high.n_vars, adata_RNA_high.n_obs - 1)
    RNA_expression = pca(adata_RNA_high, n_comps=actual_pca_comps)

    # 2. Preprocess second modality
    if dataset_name.startswith("10x"):
        adata_omics2 = adata_omics2[adata_RNA.obs_names].copy()
        adata_omics2 = Protein(adata_omics2)
        ADT_expression = adata_omics2.X
    else:
        adata_omics2 = adata_omics2[adata_RNA.obs_names].copy()
        adata_omics2.X = tfidf(adata_omics2.X)
        sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2)
        n_comps = min(60, adata_omics2.shape[1])
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps)
        ADT_expression = adata_omics2.obsm['feat']

    if scipy.sparse.issparse(ADT_expression):
        ADT_expression = ADT_expression.toarray()

    return RNA_expression, ADT_expression


# ==============================================================================
# 4. Graph Construction
# ==============================================================================

class Dual4GraphData(Data):
    """PyG Data container holding 4 graphs: RNA & Aux Similarity and Spatial graphs."""
    def __init__(self, x_RNA, x_ADT,
                 sim_edge_index_rna, sim_edge_weight_rna,
                 dist_edge_index_rna, dist_edge_weight_rna,
                 sim_edge_index_aux, sim_edge_weight_aux,
                 dist_edge_index_aux, dist_edge_weight_aux):
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


def build_4encoder_graphs(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15):
    """Construct 4 graphs: Spatial Distance Graph + RNA & Aux Cosine Similarity Graphs."""
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

    x_RNA = torch.tensor(RNA_expression, dtype=torch.float).to(device)
    x_ADT = torch.tensor(ADT_expression, dtype=torch.float).to(device)

    return Dual4GraphData(
        x_RNA=x_RNA,
        x_ADT=x_ADT,
        sim_edge_index_rna=sim_edge_index_rna,
        sim_edge_weight_rna=sim_edge_weight_rna,
        dist_edge_index_rna=dist_edge_index,
        dist_edge_weight_rna=dist_edge_weight,
        sim_edge_index_aux=sim_edge_index_aux,
        sim_edge_weight_aux=sim_edge_weight_aux,
        dist_edge_index_aux=dist_edge_index,
        dist_edge_weight_aux=dist_edge_weight
    )


# ==============================================================================
# 5. Model: All 4 Encoders are 1-Layer GCNs
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

        # 3. Arise Fusion
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


class Dual4Encoder1Layer(nn.Module):
    """Arise Model Wrapper for Total Loss (Reconstruction + Spatial Regularization + L1/L2)."""
    def __init__(self, in_channels, hidden_channels, out_channels, q, num_clusters,
                 beta=25.0, gamma=10.0, delta=1.0, dropout=0.0,
                 l1_lambda=1e-4, l2_lambda=1e-3):
        super(Dual4Encoder1Layer, self).__init__()
        self.gcn = DualGCNAll1Layer(in_channels, hidden_channels, out_channels, q, dropout)
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, out_channels))
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, x_RNA, x_ADT,
                sim_edge_index_rna, sim_edge_weight_rna,
                dist_edge_index_rna, dist_edge_weight_rna,
                sim_edge_index_aux, sim_edge_weight_aux,
                dist_edge_index_aux, dist_edge_weight_aux):
        return self.gcn(x_RNA, x_ADT,
                        sim_edge_index_rna, sim_edge_weight_rna,
                        dist_edge_index_rna, dist_edge_weight_rna,
                        sim_edge_index_aux, sim_edge_weight_aux,
                        dist_edge_index_aux, dist_edge_weight_aux)

    def compute_regularization_loss(self):
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def cosine_similarity(self, emb):
        mat = torch.matmul(emb, emb.T)
        norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
        mat = torch.div(mat, torch.matmul(norm, norm.T))
        mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
        mat = mat - torch.diag_embed(torch.diag(mat))
        return mat

    def spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight):
        num_nodes = emb.size(0)
        graph_nei = torch.sparse_coo_tensor(dist_edge_index, torch.ones_like(dist_edge_weight),
                                            size=(num_nodes, num_nodes)).to_dense()
        graph_neg = 1 - graph_nei
        sim_mat = torch.sigmoid(self.cosine_similarity(emb))

        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1 - sim_mat + 1e-10)).mean()

        return -(neigh_loss + neg_loss) / 2

    def compute_losses(self, x_RNA, x_ADT, x_sim, x_dist, aux_s, aux_d,
                       fused_rna, fused_aux, fused_pro, combined_raw):
        
        # 1. Reconstruction Loss (from joint embedding)
        l_rec_rna = F.mse_loss(x_RNA, self.gcn.reconstruct(fused_pro))
        l_rec_aux = F.mse_loss(x_ADT, self.gcn.reconstruct2(fused_pro))
        
        # 2. Correspondence Loss (Cycle-Consistency)
        # RNA Correspondence: Decode fused_rna to Aux space, re-encode through Aux encoders
        recon_aux_from_rna = self.gcn.reconstruct2(fused_rna)
        re_aux_s = self.gcn.aux_sim(recon_aux_from_rna, self.gcn_input.sim_edge_index_aux, self.gcn_input.sim_edge_weight_aux)
        re_aux_d = self.gcn.aux_dist(recon_aux_from_rna, self.gcn_input.dist_edge_index_aux, self.gcn_input.dist_edge_weight_aux)
        fused_rna_across = self.gcn.fusion_layer_aux(torch.cat([re_aux_s, re_aux_d], dim=1))
        l_corr_rna = F.mse_loss(fused_rna, fused_rna_across)
        
        # Aux Correspondence: Decode fused_aux to RNA space, re-encode through RNA encoders
        recon_rna_from_aux = self.gcn.reconstruct(fused_aux)
        re_rna_s = self.gcn.rna_sim(recon_rna_from_aux, self.gcn_input.sim_edge_index_rna, self.gcn_input.sim_edge_weight_rna)
        re_rna_d = self.gcn.rna_dist(recon_rna_from_aux, self.gcn_input.dist_edge_index_rna, self.gcn_input.dist_edge_weight_rna)
        fused_aux_across = self.gcn.fusion_layer1(torch.cat([re_rna_s, re_rna_d], dim=1))
        l_corr_aux = F.mse_loss(fused_aux, fused_aux_across)

        reg_loss = self.compute_regularization_loss()

        total_loss = (self.beta * (l_rec_rna + l_rec_aux) +
                      self.gamma * (l_corr_rna + l_corr_aux) +
                      self.delta * reg_loss)

        return total_loss, (l_rec_rna + l_rec_aux)


# ==============================================================================
# 6. Training & Evaluation Engine
# ==============================================================================

def cluster_embeddings(embeddings, num_clusters, random_state=42):
    """KMeans clustering."""
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return kmeans.fit_predict(embeddings)


def evaluate_model(model, data):
    """Return fused representations."""
    model.eval()
    with torch.no_grad():
        x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index_rna, data.sim_edge_weight_rna,
            data.dist_edge_index_rna, data.dist_edge_weight_rna,
            data.sim_edge_index_aux, data.sim_edge_weight_aux,
            data.dist_edge_index_aux, data.dist_edge_weight_aux
        )
    return fused_pro.cpu().numpy(), fused_rna.cpu().numpy(), fused_aux.cpu().numpy()


def train_model(model, data, epochs=350, lr=1e-3, num_clusters=10, true_labels=None, verbose=True):
    """
    Arise Training Loop tracking per-epoch Silhouette score, ARI, and Best Silhouette Epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    combined_raw = torch.cat([data.x_RNA, data.x_ADT], dim=1)

    best_sil = -1.0
    best_epoch = 0
    best_ari = 0.0
    best_embeddings = None
    best_labels = None

    loss_history = []
    epoch_sil_history = []
    epoch_ari_history = []

    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        x_sim, x_dist, aux_s, aux_d, fused_rna, fused_aux, fused_pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index_rna, data.sim_edge_weight_rna,
            data.dist_edge_index_rna, data.dist_edge_weight_rna,
            data.sim_edge_index_aux, data.sim_edge_weight_aux,
            data.dist_edge_index_aux, data.dist_edge_weight_aux
        )

        model.gcn_input = data
        loss, l_rec = model.compute_losses(
            data.x_RNA, data.x_ADT,
            x_sim, x_dist, aux_s, aux_d,
            fused_rna, fused_aux, fused_pro,
            combined_raw
        )

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        embeddings, _, _ = evaluate_model(model, data)
        pred_labels = cluster_embeddings(embeddings, num_clusters, random_state=42)
        sil = silhouette_score(embeddings, pred_labels)
        epoch_sil_history.append(sil)

        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, pred_labels)
            epoch_ari_history.append(ari)
        else:
            ari = None

        if sil > best_sil:
            best_sil = sil
            best_epoch = epoch + 1
            best_ari = ari if ari is not None else 0.0
            best_embeddings = embeddings.copy()
            best_labels = pred_labels.copy()

        postfix = {'loss': f'{loss.item():.4f}', 'sil': f'{sil:.4f}'}
        if ari is not None:
            postfix['ari'] = f'{ari:.4f}'
        postfix['best_sil'] = f'{best_sil:.4f} (ep {best_epoch})'
        pbar.set_postfix(postfix)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            ari_str = f" | ARI: {ari:.4f}" if ari is not None else ""
            tqdm.write(f"Epoch {epoch + 1:4d}/{epochs} | Total Loss: {loss.item():.4f} | Recon Loss: {l_rec.item():.4f} | Silhouette: {sil:.4f}{ari_str} | Best Sil: {best_sil:.4f} (Epoch {best_epoch})")

    last_sil = epoch_sil_history[-1] if epoch_sil_history else 0.0
    last_ari = epoch_ari_history[-1] if epoch_ari_history else 0.0

    if true_labels is not None and best_labels is not None:
        final_ari = adjusted_rand_score(true_labels, best_labels)
        final_nmi = normalized_mutual_info_score(true_labels, best_labels)
    else:
        final_ari, final_nmi = 0.0, 0.0

    if verbose:
        print("\n" + "=" * 65)
        print(" MODEL TRAINING FINISHED - SUMMARY ".center(65, "="))
        print("=" * 65)
        print(f"Final Epoch Total Loss        : {loss.item():.4f}")
        print(f"Last Epoch Silhouette Score   : {last_sil:.4f}")
        if true_labels is not None:
            print(f"Last Epoch ARI                : {last_ari:.4f}")
        print("-" * 65)
        print(f"Best Silhouette Score         : {best_sil:.4f} (Achieved at Epoch {best_epoch})")
        if true_labels is not None:
            print(f"ARI at Best Silhouette Epoch  : {final_ari:.4f}")
            print(f"NMI at Best Silhouette Epoch  : {final_nmi:.4f}")
        print("=" * 65 + "\n")

    return {
        'model': model,
        'best_embeddings': best_embeddings,
        'best_labels': best_labels,
        'best_sil': best_sil,
        'best_epoch': best_epoch,
        'best_ari': final_ari,
        'best_nmi': final_nmi,
        'last_sil': last_sil,
        'last_ari': last_ari,
        'loss_history': loss_history,
        'epoch_sil_history': epoch_sil_history,
        'epoch_ari_history': epoch_ari_history
    }


def plot_training_curves(training_results, dataset_name="Dataset", save_path=None, show=True):
    """
    Plot Loss Curve, Silhouette Score Curve, and ARI Curve across training epochs.

    Parameters:
    -----------
    training_results : dict
        Dictionary returned by train_model containing 'loss_history',
        'epoch_sil_history', 'epoch_ari_history', 'best_epoch', 'best_sil', and 'best_ari'.
    dataset_name : str
        Name of the dataset for plot titles.
    save_path : str, optional
        Path to save the generated figure.
    show : bool
        Whether to display the plot.
    """
    epochs_range = range(1, len(training_results['loss_history']) + 1)
    has_ari = len(training_results.get('epoch_ari_history', [])) > 0

    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Loss Curve
    axes[0].plot(epochs_range, training_results['loss_history'], color='#1f77b4', linewidth=2.0, label='Total Loss')
    axes[0].set_title(f'Loss Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True)

    # 2. Silhouette Score Curve
    axes[1].plot(epochs_range, training_results['epoch_sil_history'], color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    best_ep = training_results.get('best_epoch', 0)
    best_sil = training_results.get('best_sil', 0.0)
    if best_ep > 0:
        axes[1].axvline(x=best_ep, color='#d62728', linestyle='--', linewidth=1.5, label=f'Best Sil @ Ep {best_ep} ({best_sil:.4f})')
        axes[1].scatter([best_ep], [best_sil], color='#d62728', s=60, zorder=5)
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
    output_dir="results_spatial_arise_sgloss",
    show=False
):
    """
    Generate and save complete visualizations:
    1. Training Curves (Loss, Silhouette, ARI)
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
    sil = results['best_sil']
    ari = results.get('best_ari', 0.0)

    # Ensure adata has necessary annotations
    if true_labels is not None:
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    elif 'ground_truth' not in adata_RNA.obs:
        adata_RNA.obs['ground_truth'] = pd.Categorical(best_labels.astype(str))

    adata_RNA.obsm['Arise_1Layer'] = best_embeddings
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
        axes[1].set_title(f'Arise 4-Encoder 1-Layer Domains (ARI: {ari:.4f})', fontsize=12, fontweight='bold')
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

    # 3. 🎨 Compute UMAP on Arise 1-Layer joint embeddings
    sc.pp.neighbors(adata_RNA, use_rep='Arise_1Layer')
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
# 7. Experiment Runner (Python API)
# ==============================================================================

def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    n_seeds: Optional[int] = None,
    rna_pca_comps: int = 60,
    epochs: int = 350,
    lr: float = 1e-3,
    beta: float = 25.0,
    gamma: float = 10.0,
    delta: float = 1.0,
    hidden_dim: int = 512,
    out_dim: int = 64,
    dropout: float = 0.0,
    device: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    output_dir: str = "results_spatial_arise_sgloss",
    data_dir: str = "data"
):
    """
    Run 4-Encoder 1-Layer Arise (with RNA PCA) experiments.
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
    print(" 🚀 STARTING 4-ENCODER 1-LAYER (RNA PCA) ARISE EXPERIMENT ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets     : {dataset_names}")
    print(f"• Seeds ({len(run_seeds)})   : {run_seeds}")
    print(f"• RNA PCA Comps: {rna_pca_comps}")
    print(f"• Epochs       : {epochs} | LR: {lr}")
    print(f"• Loss Weights : Beta={beta}, Gamma={gamma}, Delta={delta}")
    print(f"• Dimensions   : Hidden={hidden_dim}, Out={out_dim} | Device: {device_str}")
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

            # Initialize 1-Layer 4-Encoder Model
            model = Dual4Encoder1Layer(
                in_channels=input_dim,
                hidden_channels=hidden_dim,
                out_channels=out_dim,
                q=adt_dim,
                num_clusters=num_clusters,
                beta=beta,
                gamma=gamma,
                delta=delta,
                dropout=dropout
            ).to(dev)

            # Train
            result = train_model(
                model=model,
                data=data,
                epochs=epochs,
                lr=lr,
                num_clusters=num_clusters,
                true_labels=true_labels,
                verbose=True
            )

            best_embeddings = result['best_embeddings']
            best_labels = result['best_labels']
            best_sil = result['best_sil']
            best_epoch = result['best_epoch']

            # Compute Final Metrics
            ari = adjusted_rand_score(true_labels, best_labels)
            nmi = normalized_mutual_info_score(true_labels, best_labels)
            ami = adjusted_mutual_info_score(true_labels, best_labels)
            chi = calinski_harabasz_score(best_embeddings, best_labels)
            dbi = davies_bouldin_score(best_embeddings, best_labels)

            last_sil = result['last_sil']
            last_ari = result['last_ari']

            print(f"Summary for {dataset_name} | Seed: {seed}")
            print(f"• Best Silhouette : {best_sil:.4f} (Epoch {best_epoch}) | ARI @ Best: {ari:.4f} | NMI: {nmi:.4f}")
            print(f"• Last Silhouette : {last_sil:.4f} (Epoch {epochs})     | Last ARI   : {last_ari:.4f}")

            # Plot and save all visualizations if requested
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
                'Best_Sil_Epoch': best_epoch,
                'Best_ARI': ari,
                'Best_NMI': nmi,
                'Last_Silhouette': last_sil,
                'Last_ARI': last_ari,
                'AMI': ami,
                'CHI': chi,
                'DBI': dbi,
                'RNA_PCA_Comps': rna_pca_comps,
                'no_cluster': num_clusters
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)

        # Save per-dataset results
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"Arise4Encoder1Layer_{dataset_name}_results.csv"), index=False)

    # Save overall summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "Arise4Encoder1Layer_all_results.csv"), index=False)
    
    summary_cols = ['Best_ARI', 'Best_Silhouette', 'Last_ARI', 'Last_Silhouette', 'Best_NMI']
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
# 8. Command-Line Interface (CLI)
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="4-Encoder 1-Layer Arise (RNA PCA) Multi-Omics Runner",
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
    parser.add_argument('--epochs', type=int, default=350, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--beta', type=float, default=25.0, help="Reconstruction loss weight")
    parser.add_argument('--gamma', type=float, default=10.0, help="Spatial regularization weight")
    parser.add_argument('--delta', type=float, default=1.0, help="Weight decay regularization weight")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--out_dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--device', type=str, default=None, help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--visualize', action='store_true', default=True, help="Generate and save all plots (Curves, Spatial, UMAP, Violin)")
    parser.add_argument('--show_plots', action='store_true', default=False, help="Display plots interactively")
    parser.add_argument('--output_dir', type=str, default='results_spatial_arise_sgloss', help="Directory to save CSV results and plots")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save/load datasets")

    cli_args = parser.parse_args()

    run_experiment(
        datasets=cli_args.datasets if len(cli_args.datasets) > 1 or cli_args.datasets[0] != 'all' else 'all',
        seeds=cli_args.seeds,
        n_seeds=cli_args.n_seeds,
        rna_pca_comps=cli_args.rna_pca_comps,
        epochs=cli_args.epochs,
        lr=cli_args.lr,
        beta=cli_args.beta,
        gamma=cli_args.gamma,
        delta=cli_args.delta,
        hidden_dim=cli_args.hidden_dim,
        out_dim=cli_args.out_dim,
        dropout=cli_args.dropout,
        device=cli_args.device,
        visualize=cli_args.visualize,
        show_plots=cli_args.show_plots,
        output_dir=cli_args.output_dir,
        data_dir=cli_args.data_dir
    )
