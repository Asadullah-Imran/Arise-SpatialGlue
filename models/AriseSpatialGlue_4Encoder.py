# ==============================================================================
# AriseSpatialGlue_4Encoder.py
#
# Arise-Based 4-Encoder Framework:
# - Encoders: 4 GCN encoders
#     * Encoder 1 (RNA): 2-layer GCN on RNA Similarity Graph
#     * Encoder 2 (RNA): 2-layer GCN on Spatial Distance Graph
#     * Encoder 3 (Aux): 1-layer GCN on Aux Similarity Graph
#     * Encoder 4 (Aux): 1-layer GCN on Spatial Distance Graph
# - Fusion: Arise Linear & MLP concatenation-based fusion layers
# - Decoders: Arise multi-target reconstruction (RNA, Aux, Joint)
# - Loss: Arise total loss = beta * (Recon) + gamma * (Spatial Reg) + delta * (L1/L2)
# - Preprocessing & Hyperparameters: Identical to Arise
# - Per-Epoch Logging: Silhouette Score, ARI, and Best Silhouette Epoch reporting
# - Custom Input Support: Datasets, Seeds, Epochs, Hyperparameters via CLI or Python API
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
# 3. Preprocessing (Identical to Arise)
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


def pca(adata, use_reps=None, n_comps=10):
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


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs):
    """Latent Semantic Indexing for ATAC data."""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def preprocess_universal(adata_RNA, adata_omics2, dataset_name):
    """Universal preprocessing identical to Arise codebase."""
    sc.pp.filter_genes(adata_RNA, min_cells=10)
    sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.scale(adata_RNA)

    RNA_expression = adata_RNA[:, adata_RNA.var['highly_variable']].X
    if scipy.sparse.issparse(RNA_expression):
        RNA_expression = RNA_expression.toarray()

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
# 4. 4-Graph Data Structure & Graph Construction
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

    dist_edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long).to(device)
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
# 5. Model: 4-Encoder Architecture with Arise Fusion & Losses
# ==============================================================================

class DualGCN4Encoder(nn.Module):
    """4-Encoder GCN architecture with Arise-style linear fusion & reconstruction decoders."""
    def __init__(self, in_channels, hidden_channels, out_channels, q, dropout=0.0):
        super(DualGCN4Encoder, self).__init__()

        # 2 RNA Encoders (2-layer GCNs)
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.sim_conv = GCNConv(hidden_channels, out_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels)
        self.dist_conv = GCNConv(hidden_channels, out_channels)

        # 2 Auxiliary Encoders (1-layer GCNs)
        self.aux_dist = GCNConv(q, out_channels)
        self.aux_sim = GCNConv(q, out_channels)

        # Arise Linear Fusion
        self.fusion_layer1 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer_aux = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer2 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))

        # Arise Decoders
        self.dropout = dropout
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)
        self.deconv4 = nn.Linear(hidden_channels, q)
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels)

    def forward(self, x_RNA, x_ADT,
                sim_edge_index_rna, sim_edge_weight_rna,
                dist_edge_index_rna, dist_edge_weight_rna,
                sim_edge_index_aux, sim_edge_weight_aux,
                dist_edge_index_aux, dist_edge_weight_aux):

        # RNA Stream
        xs = F.relu(self.x_RNA1(x_RNA, sim_edge_index_rna, sim_edge_weight_rna))
        xs = F.dropout(xs, self.dropout, training=self.training)
        x_sim = self.sim_conv(xs, sim_edge_index_rna, sim_edge_weight_rna)

        xd = F.relu(self.x_RNA2(x_RNA, dist_edge_index_rna, dist_edge_weight_rna))
        xd = F.dropout(xd, self.dropout, training=self.training)
        x_dist = self.dist_conv(xd, dist_edge_index_rna, dist_edge_weight_rna)

        # Aux Stream
        aux_d = self.aux_dist(x_ADT, dist_edge_index_aux, dist_edge_weight_aux)
        aux_s = self.aux_sim(x_ADT, sim_edge_index_aux, sim_edge_weight_aux)

        # Fusion
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


class Dual4Encoder(nn.Module):
    """Arise Model Wrapper for Total Loss (Reconstruction + Spatial Regularization + L1/L2)."""
    def __init__(self, in_channels, hidden_channels, out_channels, q, num_clusters,
                 beta=25.0, gamma=10.0, delta=1.0, dropout=0.0,
                 l1_lambda=1e-4, l2_lambda=1e-3):
        super(Dual4Encoder, self).__init__()
        self.gcn = DualGCN4Encoder(in_channels, hidden_channels, out_channels, q, dropout)
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
        l_rec = F.mse_loss(combined_raw, self.gcn.reconstruct3(fused_pro))
        l_sim = F.mse_loss(x_RNA, self.gcn.reconstruct(x_sim))
        l_dist = F.mse_loss(x_RNA, self.gcn.reconstruct(x_dist))
        l_aux_s = F.mse_loss(x_ADT, self.gcn.reconstruct2(aux_s))
        l_aux_d = F.mse_loss(x_ADT, self.gcn.reconstruct2(aux_d))

        l_spatial = self.spatial_regularization_loss(
            fused_rna,
            dist_edge_index=self.gcn_input.dist_edge_index_rna,
            dist_edge_weight=self.gcn_input.dist_edge_weight_rna
        )

        reg_loss = self.compute_regularization_loss()

        total_loss = (self.beta * (l_rec + l_sim + l_dist + l_aux_s + l_aux_d) +
                      self.gamma * l_spatial +
                      self.delta * reg_loss)

        return total_loss, l_rec


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

        # Evaluate epoch clustering representation
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

        # Update progress bar
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


# ==============================================================================
# 7. Experiment Runner (Python API)
# ==============================================================================

def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    n_seeds: Optional[int] = None,
    epochs: int = 350,
    lr: float = 1e-3,
    beta: float = 25.0,
    gamma: float = 10.0,
    delta: float = 1.0,
    hidden_dim: int = 512,
    out_dim: int = 64,
    dropout: float = 0.0,
    device: Optional[str] = None,
    save_curves: bool = True,
    output_dir: str = "results",
    data_dir: str = "data"
):
    """
    Run 4-Encoder Arise experiments on specified datasets and seeds.

    Args:
        datasets: Dataset name(s), index/indices (0-5), or 'all'.
        seeds: List of random seeds (defaults to Arise 20 seeds).
        n_seeds: Number of seeds to use from the seed list.
        epochs: Number of training epochs (default: 350).
        lr: Learning rate (default: 1e-3).
        beta: Reconstruction loss weight (default: 25.0).
        gamma: Spatial regularization weight (default: 10.0).
        delta: Weight decay regularization weight (default: 1.0).
        hidden_dim: Hidden layer size (default: 512).
        out_dim: Output embedding dimension (default: 64).
        dropout: Dropout rate (default: 0.0).
        device: 'cuda', 'cuda:0', or 'cpu'.
        output_dir: Directory to save results CSV.
        data_dir: Directory to store downloaded datasets.
    """
    if device is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = device
    dev = torch.device(device_str)

    # Resolve datasets
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
                raise ValueError(f"Invalid dataset index {d}. Must be between 0 and {len(DATASET_LIST)-1}.")
        elif isinstance(d, str):
            if d in DATASET_REGISTRY:
                dataset_names.append(d)
            elif d.isdigit() and 0 <= int(d) < len(DATASET_LIST):
                dataset_names.append(DATASET_LIST[int(d)])
            else:
                raise ValueError(f"Dataset '{d}' not recognized. Choices: {DATASET_LIST}")

    # Resolve seeds
    if seeds is None:
        run_seeds = DEFAULT_SEEDS
    else:
        run_seeds = list(seeds)

    if n_seeds is not None and n_seeds > 0:
        run_seeds = run_seeds[:n_seeds]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "=" * 80)
    print(" 🚀 STARTING 4-ENCODER ARISE EXPERIMENT RUNNER ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets     : {dataset_names}")
    print(f"• Seeds ({len(run_seeds)})   : {run_seeds}")
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
        num_clusters = len(np.unique(true_labels))
        print(f"Dataset: {dataset_name} | Spots: {adata_RNA.n_obs} | Clusters: {num_clusters}")

        # Preprocessing
        RNA_expr, ADT_expr = preprocess_universal(adata_RNA, adata_omics2, dataset_name)
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

            # Initialize Model
            model = Dual4Encoder(
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

            # Plot and save curves if requested
            if save_curves:
                curve_save_dir = os.path.join(output_dir, "curves")
                os.makedirs(curve_save_dir, exist_ok=True)
                curve_path = os.path.join(curve_save_dir, f"Arise4Encoder_{dataset_name}_seed_{seed}_curves.png")
                plot_training_curves(result, dataset_name=f"{dataset_name} (Seed {seed})", save_path=curve_path, show=False)

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
                'no_cluster': num_clusters
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)

        # Save per-dataset results
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"Arise4Encoder_{dataset_name}_results.csv"), index=False)

    # Save overall summary
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "Arise4Encoder_all_results.csv"), index=False)
    
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
        description="4-Encoder Arise Multi-Omics Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset & Seed Inputs
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

    # Training & Hyperparameter Inputs
    parser.add_argument('--epochs', type=int, default=350, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--beta', type=float, default=25.0, help="Reconstruction loss weight")
    parser.add_argument('--gamma', type=float, default=10.0, help="Spatial regularization weight")
    parser.add_argument('--delta', type=float, default=1.0, help="Weight decay regularization weight")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension")
    parser.add_argument('--out_dim', type=int, default=64, help="Embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--device', type=str, default=None, help="'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save CSV results")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save/load datasets")

    cli_args = parser.parse_args()

    # Pass CLI arguments to run_experiment
    run_experiment(
        datasets=cli_args.datasets if len(cli_args.datasets) > 1 or cli_args.datasets[0] != 'all' else 'all',
        seeds=cli_args.seeds,
        n_seeds=cli_args.n_seeds,
        epochs=cli_args.epochs,
        lr=cli_args.lr,
        beta=cli_args.beta,
        gamma=cli_args.gamma,
        delta=cli_args.delta,
        hidden_dim=cli_args.hidden_dim,
        out_dim=cli_args.out_dim,
        dropout=cli_args.dropout,
        device=cli_args.device,
        output_dir=cli_args.output_dir,
        data_dir=cli_args.data_dir
    )
